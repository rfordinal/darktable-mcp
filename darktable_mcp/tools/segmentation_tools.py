"""Bridge from the MCP server process to the SAM2 segmentation sidecar (T2.1),
with an in-process OpenCV GrabCut fallback (see local_segment.py) for the
common case where the sidecar isn't installed/reachable.

Selection order, every call:
    1. Try the SAM2 sidecar subprocess (`_run_sidecar`) -- if it is
       configured (env vars / co-located dev venv) AND runs successfully,
       its (higher-quality) polygon wins.
    2. If the sidecar is not configured, not reachable, times out, or
       fails in any other way, fall back to the local GrabCut segmenter
       (`local_segment.segment_grabcut`) -- zero-install, bundled in this
       package's own venv (opencv-python-headless + numpy are ordinary
       pyproject dependencies, not optional extras).
    3. Only if BOTH fail is a SegmentationServiceError raised to the
       caller (mask_object) -- the error message includes why each one
       failed.
Either path's result carries `"backend"` ("sam2" or "grabcut") so
mask_object can tell the caller which quality tier was actually used.

The sidecar (`darktable-mcp/sidecar/segment.py`) lives in its OWN uv venv
(Python 3.12 + torch/sam2), separate from both this package's `.venv` and
the darktable process. darktable itself runs inside the docker bridge
container (docker/run-dt-bridge.sh) which has no Python/torch at all -- only
`/opt/darktable` (read-only) and the bind-mounted `/run` cache dir. So
segmentation cannot run "inside darktable" by construction; it doesn't need
to. This MCP server process runs on the HOST (it's the thing polling
request-*.json/response-*.json against the container), so the natural, and
only practical, place to invoke the sidecar is right here: a subprocess to
the sidecar's own venv Python, on the host, passing it a preview PNG that
`get_preview` already wrote to a host-readable path (server.py's
`_remap_bridge_path` handles the container->host path swap for that PNG
before it ever reaches this module). No new IPC channel, no code sharing
across venvs -- just `subprocess.run([sidecar_venv_python, segment.py, ...])`
and parse its stdout JSON. This mirrors how CameraTools/CLIWrapper already
shell out to external tools (gphoto2, darktable-cli) elsewhere in this repo.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils.errors import SegmentationServiceError
from . import local_segment

logger = logging.getLogger(__name__)

# darktable-mcp/darktable_mcp/tools/segmentation_tools.py -> darktable-mcp/
_MCP_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_DIR = _MCP_ROOT / "sidecar"
SIDECAR_SCRIPT = SIDECAR_DIR / "segment.py"

# The sidecar's own venv (see sidecar/README.md "Enabling ... from scratch":
# `uv venv --python 3.12 .venv`), NOT darktable-mcp/.venv and NOT the system
# python3. This package deliberately does NOT bundle the sidecar (torch CPU
# + the 149MB checkpoint are too heavy / the wrong shape for this .deb -- see
# dist/INSTALL-sidecar.md), so on a packaged install SIDECAR_DIR normally
# does not exist at all and the two env vars below are the only way to point
# this process at wherever the sidecar venv actually got set up:
#   DARKTABLE_MCP_SIDECAR_PYTHON  -- path to the sidecar venv's python3.12
#   DARKTABLE_MCP_SIDECAR_SEGMENT -- path to the sidecar's segment.py
# Falls back to the co-located dev-checkout layout (sidecar/.venv/...) when
# unset, which is what the in-repo spike/tests use. Also overridable per-call
# via run_segmentation(sidecar_python=..., sidecar_script=...) (e.g. tests
# pointing at a nonexistent path to simulate "sidecar down" -- ACCEPTANCE
# T2.3-S2 / T2.5 "no sidecar configured").
ENV_SIDECAR_PYTHON = "DARKTABLE_MCP_SIDECAR_PYTHON"
ENV_SIDECAR_SEGMENT = "DARKTABLE_MCP_SIDECAR_SEGMENT"
DEFAULT_SIDECAR_PYTHON = SIDECAR_DIR / ".venv" / "bin" / "python3.12"
DEFAULT_CHECKPOINT = SIDECAR_DIR / "checkpoints" / "sam2.1_hiera_tiny.pt"
# Hydra config identifier bundled inside the installed `sam2` package (NOT a
# filesystem path relative to cwd) -- see sidecar/README.md CLI example.
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

DEFAULT_TIMEOUT = 60.0


def _sidecar_python(override: Optional[str] = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get(ENV_SIDECAR_PYTHON)
    if env:
        return Path(env)
    return DEFAULT_SIDECAR_PYTHON


def _sidecar_script(override: Optional[str] = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get(ENV_SIDECAR_SEGMENT)
    if env:
        return Path(env)
    return SIDECAR_SCRIPT


def run_segmentation(
    image_path: str,
    *,
    points: Optional[List[Dict[str, Any]]] = None,
    box: Optional[Dict[str, float]] = None,
    label: Optional[str] = None,
    backend: str = "sam2",
    checkpoint: Optional[str] = None,
    model_cfg: Optional[str] = None,
    sidecar_python: Optional[str] = None,
    sidecar_script: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    allow_grabcut_fallback: bool = True,
) -> Dict[str, Any]:
    """Segment `image_path` for the given prompt, preferring the SAM2
    sidecar and falling back to the in-process GrabCut segmenter (see
    module docstring for the full selection order).

    Returns the (SAM2 or GrabCut) result's JSON contract verbatim:
        {"polygon": [{"x","y"}, ...], "bbox": {...}, "score": float|None,
         "num_points_raw", "num_points_simplified", "backend", "image_size"}
    `backend` is "sam2" or "grabcut" depending on which one actually ran --
    callers should surface it so the caller/user knows the quality tier.

    Raises SegmentationServiceError only if BOTH the sidecar AND the
    GrabCut fallback fail (or if `allow_grabcut_fallback=False` and only
    the sidecar was tried) -- callers must treat that as "no polygon, no
    side effects", never a half-populated result.

    `allow_grabcut_fallback=False` disables the fallback (e.g. a caller
    that specifically wants to test/force sidecar-only behavior); the
    default is True so mask_object always gets *something* zero-install.
    """
    if not points and not box and not label:
        raise SegmentationServiceError(
            "mask_object needs at least one of points/box/label to prompt "
            "the segmentation sidecar -- pick a point or box on the subject "
            "from a get_preview image first."
        )

    sidecar_error: Optional[Exception] = None
    try:
        return _run_sidecar(
            image_path,
            points=points,
            box=box,
            label=label,
            backend=backend,
            checkpoint=checkpoint,
            model_cfg=model_cfg,
            sidecar_python=sidecar_python,
            sidecar_script=sidecar_script,
            timeout=timeout,
        )
    except SegmentationServiceError as exc:
        sidecar_error = exc
        if not allow_grabcut_fallback:
            raise

    logger.info(
        "SAM2 sidecar unavailable (%s); falling back to in-process GrabCut",
        sidecar_error,
    )
    try:
        return local_segment.segment_grabcut(image_path, points=points, box=box, label=label)
    except Exception as grabcut_error:  # noqa: BLE001 - report both failures verbatim
        raise SegmentationServiceError(
            "segmentation unavailable: SAM2 sidecar failed "
            f"({sidecar_error}); GrabCut fallback also failed "
            f"({grabcut_error})"
        ) from grabcut_error


def _run_sidecar(
    image_path: str,
    *,
    points: Optional[List[Dict[str, Any]]] = None,
    box: Optional[Dict[str, float]] = None,
    label: Optional[str] = None,
    backend: str = "sam2",
    checkpoint: Optional[str] = None,
    model_cfg: Optional[str] = None,
    sidecar_python: Optional[str] = None,
    sidecar_script: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Invoke `sidecar/segment.py` as a subprocess in its own venv.

    Returns the sidecar's JSON contract verbatim (see run_segmentation's
    docstring). Raises SegmentationServiceError on ANY failure (venv/binary
    missing, non-zero exit, timeout, malformed stdout) -- run_segmentation
    catches this to try the GrabCut fallback.
    """
    if not points and not box and not label:
        raise SegmentationServiceError(
            "mask_object needs at least one of points/box/label to prompt "
            "the segmentation sidecar -- pick a point or box on the subject "
            "from a get_preview image first."
        )

    python = _sidecar_python(sidecar_python)
    script = _sidecar_script(sidecar_script)
    if not python.is_file():
        raise SegmentationServiceError(
            "segmentation sidecar not installed/configured: sidecar python "
            f"not found at {python}. Install the optional SAM2 sidecar (see "
            "dist/INSTALL-sidecar.md) and set DARKTABLE_MCP_SIDECAR_PYTHON "
            "(and DARKTABLE_MCP_SIDECAR_SEGMENT if it isn't at the default "
            "in-repo location), or pass sidecar_python explicitly. "
            "add_path_mask and every other tool work fine without it -- "
            "only mask_object's auto-segmentation needs the sidecar."
        )
    if not script.is_file():
        raise SegmentationServiceError(
            f"segmentation sidecar not installed/configured: {script} not "
            "found. Install the optional SAM2 sidecar (see "
            "dist/INSTALL-sidecar.md) and set DARKTABLE_MCP_SIDECAR_SEGMENT, "
            "or pass sidecar_script explicitly."
        )

    cmd: List[str] = [str(python), str(script), "--image", str(image_path)]
    for p in points or []:
        x, y = p["x"], p["y"]
        lbl = p.get("label", 1)
        cmd += ["--point", f"{x},{y},{lbl}"]
    if box:
        cmd += ["--box", f"{box['x']},{box['y']},{box['w']},{box['h']}"]
    if label:
        cmd += ["--label", label]

    cmd += ["--backend", backend]
    if backend == "sam2":
        # FIX: default checkpoint next to the ACTUAL configured segment.py
        # (script.parent), not the hardcoded dev-checkout SIDECAR_DIR. The
        # dev-checkout default (DEFAULT_CHECKPOINT) only happens to be
        # right when `script` IS the co-located dev sidecar; any real
        # install (env vars pointing elsewhere -- e.g. darktable-mcp
        # install-sidecar's own default ~/.local/share/darktable-mcp/
        # sidecar/) previously silently looked in the wrong place and
        # always fell through to "sidecar failed", which is straightforward
        # to miss because the GrabCut fallback (see run_segmentation) masks
        # it with a working-but-lower-quality result instead of a loud error.
        cmd += ["--checkpoint", checkpoint or str(script.parent / "checkpoints" / "sam2.1_hiera_tiny.pt")]
        cmd += ["--model-cfg", model_cfg or DEFAULT_MODEL_CFG]

    logger.debug("segmentation sidecar cmd: %s", cmd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(script.parent),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise SegmentationServiceError(
            f"segmentation service unavailable: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SegmentationServiceError(
            f"segmentation service unavailable: sidecar timed out after "
            f"{timeout}s"
        ) from exc

    if proc.returncode != 0:
        raise SegmentationServiceError(
            "segmentation service unavailable: sidecar exited with code "
            f"{proc.returncode}: {proc.stderr.strip()[-2000:] or '(no stderr)'}"
        )

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise SegmentationServiceError(
            f"segmentation service returned invalid JSON: {exc}; "
            f"stdout={proc.stdout[:500]!r}"
        ) from exc

    if not isinstance(result, dict) or "polygon" not in result:
        raise SegmentationServiceError(
            f"segmentation service returned unexpected shape: {result!r}"
        )
    return result
