"""segment.py -- darktable-mcp segmentation sidecar (PLAN.md T2.1).

Turns a point/box (and optionally free-text label, best-effort) prompt into a
normalized polygon contour ready to hand to ``darktable.develop.add_path_mask``
(T2.2). This module is a standalone service: it does not import or touch any
darktable C code, and has no dependency on the rest of ``darktable_mcp``.

Pipeline
--------
    prompt (points / box / label)
        -> SegmentationModel.predict()   binary mask, HxW bool, plus a score
        -> largest external contour      cv2.findContours
        -> polygon simplification        cv2.approxPolyDP (Douglas-Peucker),
                                          binary-searched to ~10-30 nodes
        -> normalization                 pixel coords -> 0..1 by image size
        -> {polygon, bbox, score, ...}

Multi-part / holes
-------------------
``add_path_mask`` (and darktable's ``DT_MASKS_PATH`` form in general) wants a
single closed contour. If the predicted mask has holes or several disjoint
blobs, this module keeps only the *largest-area external* contour and drops
the rest. That is a deliberate simplification, not a bug -- document it to
callers (T2.3) so "brighten her face" over a mask with an earring-shaped hole
still gets one clean outline.

Model backend is swappable -- see ``load_model()`` below, which is the single
function T2.3 (or a future add-on) needs to touch to point at a different
segmenter or a different SAM2 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image

# --------------------------------------------------------------------------
# Prompt / result data model
# --------------------------------------------------------------------------


@dataclass
class Point:
    """A single prompt point, normalized 0..1, with a SAM-style foreground/
    background label (1 = include, 0 = exclude)."""

    x: float
    y: float
    label: int = 1

    def to_px(self, width: int, height: int) -> tuple[float, float]:
        return self.x * width, self.y * height


@dataclass
class Box:
    """A prompt box, normalized 0..1, top-left + width/height."""

    x: float
    y: float
    w: float
    h: float

    def to_px_xyxy(self, width: int, height: int) -> tuple[float, float, float, float]:
        x0 = self.x * width
        y0 = self.y * height
        x1 = (self.x + self.w) * width
        y1 = (self.y + self.h) * height
        return x0, y0, x1, y1


@dataclass
class SegmentationResult:
    """Raw model output before the contour/simplify/normalize pipeline."""

    mask: np.ndarray  # HxW bool
    score: Optional[float] = None
    raw: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Segmentation model interface (swappable backend)
# --------------------------------------------------------------------------


class SegmentationModel(ABC):
    """Interface every backend (real SAM2, stub, future MODNet, ...) must
    implement. Only ``predict`` is required downstream."""

    @abstractmethod
    def predict(
        self,
        image_rgb: np.ndarray,
        points: Sequence[Point] = (),
        box: Optional[Box] = None,
        label: Optional[str] = None,
    ) -> SegmentationResult:
        """image_rgb: HxWx3 uint8. points/box are normalized 0..1 (see Point/Box).
        label: free-text hint, best-effort, may be ignored by a backend that
        has no text grounding (SAM2 itself has none -- see README)."""
        raise NotImplementedError


class SAM2Model(SegmentationModel):
    """Real backend: Meta's Segment Anything 2, image predictor, point/box
    prompts. Text prompts are NOT natively supported by SAM2 (it has no
    language tower) -- `label` is accepted for interface symmetry with the
    future orchestrator (T2.3, which will have already turned "face" into
    points/box via Claude vision before calling here) but is otherwise inert.

    Lazy-imports torch/sam2 so the rest of this module (contour/simplify/
    normalize + the stub backend) works even where SAM2 is not installed.
    """

    def __init__(self, checkpoint: str, model_cfg: str, device: str = "cpu"):
        import torch  # noqa: F401  (import guarded here on purpose)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = device
        sam2_model = build_sam2(model_cfg, checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)

    def predict(
        self,
        image_rgb: np.ndarray,
        points: Sequence[Point] = (),
        box: Optional[Box] = None,
        label: Optional[str] = None,
    ) -> SegmentationResult:
        h, w = image_rgb.shape[:2]
        self.predictor.set_image(image_rgb)

        point_coords = None
        point_labels = None
        if points:
            point_coords = np.array([p.to_px(w, h) for p in points], dtype=np.float32)
            point_labels = np.array([p.label for p in points], dtype=np.int32)

        box_xyxy = None
        if box is not None:
            box_xyxy = np.array(box.to_px_xyxy(w, h), dtype=np.float32)

        masks, scores, _logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_xyxy,
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return SegmentationResult(
            mask=masks[best].astype(bool),
            score=float(scores[best]),
            raw={"backend": "sam2", "num_candidates": len(scores)},
        )


class StubEllipseModel(SegmentationModel):
    """NOT a real segmentation model. Deterministic geometric stand-in used
    to verify the contour -> simplify -> normalize pipeline end to end when
    real SAM2 weights are unavailable/impractical (see README). It fits an
    axis-aligned ellipse mask to the prompted box (or, if only points were
    given, to a fixed-radius disc around their centroid) and returns that as
    the "mask". Score is always 1.0 (no model uncertainty to report).

    Any test asserting behaviour of *this* class is testing geometry code,
    not segmentation quality.
    """

    def __init__(self, default_radius_frac: float = 0.2):
        self.default_radius_frac = default_radius_frac

    def predict(
        self,
        image_rgb: np.ndarray,
        points: Sequence[Point] = (),
        box: Optional[Box] = None,
        label: Optional[str] = None,
    ) -> SegmentationResult:
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if box is not None:
            x0, y0, x1, y1 = box.to_px_xyxy(w, h)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            ax, ay = max((x1 - x0) / 2.0, 1.0), max((y1 - y0) / 2.0, 1.0)
        elif points:
            xs = [p.x * w for p in points if p.label == 1]
            ys = [p.y * h for p in points if p.label == 1]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            r = self.default_radius_frac * min(w, h)
            ax = ay = r
        else:
            cx, cy = w / 2.0, h / 2.0
            ax = ay = self.default_radius_frac * min(w, h)

        cv2.ellipse(
            mask,
            center=(int(round(cx)), int(round(cy))),
            axes=(int(round(ax)), int(round(ay))),
            angle=0,
            startAngle=0,
            endAngle=360,
            color=1,
            thickness=-1,
        )
        return SegmentationResult(mask=mask.astype(bool), score=1.0, raw={"backend": "stub-ellipse"})


def load_model(backend: str = "sam2", **kwargs) -> SegmentationModel:
    """The single swap-in point. Call this to get whichever backend the
    caller wants; everything downstream (contour/simplify/normalize) is
    backend-agnostic.

    backend="sam2"  -> real SAM2Model. Requires `checkpoint` + `model_cfg`
                        kwargs (or SAM2_CHECKPOINT / SAM2_MODEL_CFG env vars),
                        and the `sam2` + `torch` packages installed -- see
                        README "Enabling real SAM2".
    backend="stub"  -> StubEllipseModel, no dependencies beyond numpy/opencv.
    """
    if backend == "sam2":
        import os

        checkpoint = kwargs.get("checkpoint") or os.environ.get("SAM2_CHECKPOINT")
        model_cfg = kwargs.get("model_cfg") or os.environ.get("SAM2_MODEL_CFG")
        if not checkpoint or not model_cfg:
            raise ValueError(
                "sam2 backend requires checkpoint + model_cfg "
                "(args or SAM2_CHECKPOINT / SAM2_MODEL_CFG env vars); see README"
            )
        device = kwargs.get("device", "cpu")
        return SAM2Model(checkpoint=checkpoint, model_cfg=model_cfg, device=device)
    if backend == "stub":
        return StubEllipseModel(default_radius_frac=kwargs.get("default_radius_frac", 0.2))
    raise ValueError(f"unknown backend: {backend!r} (expected 'sam2' or 'stub')")


# --------------------------------------------------------------------------
# Mask -> contour -> simplify -> normalize pipeline (real, backend-agnostic)
# --------------------------------------------------------------------------


def largest_external_contour(mask: np.ndarray) -> np.ndarray:
    """cv2.findContours + pick the largest-area external contour. See module
    docstring: holes / secondary blobs are intentionally dropped -- a path
    mask is a single closed contour.

    Returns an (N, 2) int32 array of (x, y) pixel coordinates. Raises
    ValueError if the mask is empty (no foreground pixels)."""
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _hierarchy = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("mask has no foreground pixels; nothing to contour")
    largest = max(contours, key=cv2.contourArea)
    return largest.reshape(-1, 2).astype(np.int32)


def simplify_polygon(
    contour: np.ndarray,
    target_min: int = 10,
    target_max: int = 30,
    max_iter: int = 25,
) -> np.ndarray:
    """Douglas-Peucker simplification (cv2.approxPolyDP) with the epsilon
    binary-searched so the resulting node count lands in [target_min,
    target_max] whenever the raw contour has at least target_min points to
    begin with. Returns an (M, 2) int32 array, M in [target_min, target_max]
    (or the raw contour unchanged if it already has <= target_max points, or
    the best achieved M if target_min can't be hit -- e.g. a near-perfect
    ellipse simplifies fast and can't be forced back up to target_min without
    reintroducing raw jaggies)."""
    contour = contour.reshape(-1, 1, 2).astype(np.int32)
    n = len(contour)
    if n <= target_max:
        return contour.reshape(-1, 2)

    perimeter = cv2.arcLength(contour, True)
    lo, hi = 0.0001 * perimeter, 0.2 * perimeter
    best = contour.reshape(-1, 2)
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        approx = cv2.approxPolyDP(contour, mid, True).reshape(-1, 2)
        m = len(approx)
        if target_min <= m <= target_max:
            return approx
        if m > target_max:
            lo = mid  # too many points -> increase epsilon
        else:
            hi = mid  # too few points -> decrease epsilon
        # keep the closest-to-range candidate seen so far as a fallback
        if abs(m - target_max) < abs(len(best) - target_max) or (
            target_min <= m and len(best) < target_min
        ):
            best = approx
    return best


def normalize_polygon(points_px: np.ndarray, width: int, height: int) -> list[dict]:
    """(N,2) pixel coords -> list of {"x","y"} normalized 0..1, in order,
    ready for add_path_mask. Contour order from cv2 is already a closed ring
    (implicit closure: last point connects back to first); we do not repeat
    the first point at the end, matching darktable's own path-node list
    convention."""
    return [
        {"x": float(px / width), "y": float(py / height)}
        for px, py in points_px
    ]


def polygon_bbox(polygon_norm: list[dict]) -> dict:
    xs = [p["x"] for p in polygon_norm]
    ys = [p["y"] for p in polygon_norm]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


# --------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------


def segment(
    image_path: str,
    points: Optional[list[dict]] = None,
    box: Optional[dict] = None,
    label: Optional[str] = None,
    model: Optional[SegmentationModel] = None,
    backend: str = "sam2",
    target_min: int = 10,
    target_max: int = 30,
    **backend_kwargs,
) -> dict:
    """Main entrypoint T2.3 (or anything else) calls.

    points: list of {"x":.., "y":.., "label": 1|0}  (normalized 0..1; label
            optional, defaults to 1 = foreground click)
    box:    {"x":.., "y":.., "w":.., "h":..}         (normalized 0..1)
    label:  free-text hint, best-effort (see SAM2Model docstring -- SAM2 has
            no text grounding; the real translation "face" -> points/box is
            T2.3's job using Claude vision, upstream of this call)
    model:  pass a pre-loaded SegmentationModel to reuse across calls
            (skips reloading weights); otherwise one is built via
            load_model(backend, **backend_kwargs) for this call only.

    Returns:
        {
          "polygon": [{"x":.., "y":..}, ...],   # ~target_min..target_max nodes
          "bbox": {"x":.., "y":.., "w":.., "h":..},
          "score": float | null,
          "num_points_raw": int,
          "num_points_simplified": int,
          "backend": str,
          "image_size": {"w": int, "h": int},
        }
    """
    if not points and not box and not label:
        raise ValueError("segment() needs at least one of points, box, label")

    pil_img = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_img)
    height, width = image_rgb.shape[:2]

    pts = [Point(**p) for p in (points or [])]
    bx = Box(**box) if box else None

    if model is None:
        model = load_model(backend=backend, **backend_kwargs)

    result = model.predict(image_rgb, points=pts, box=bx, label=label)

    raw_contour = largest_external_contour(result.mask)
    simplified = simplify_polygon(raw_contour, target_min=target_min, target_max=target_max)
    polygon_norm = normalize_polygon(simplified, width, height)
    bbox_norm = polygon_bbox(polygon_norm)

    return {
        "polygon": polygon_norm,
        "bbox": bbox_norm,
        "score": result.score,
        "num_points_raw": int(len(raw_contour)),
        "num_points_simplified": int(len(simplified)),
        "backend": result.raw.get("backend", backend),
        "image_size": {"w": int(width), "h": int(height)},
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_point(s: str) -> dict:
    parts = s.split(",")
    x, y = float(parts[0]), float(parts[1])
    label = int(parts[2]) if len(parts) > 2 else 1
    return {"x": x, "y": y, "label": label}


def _parse_box(s: str) -> dict:
    x, y, w, h = (float(v) for v in s.split(","))
    return {"x": x, "y": y, "w": w, "h": h}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="path to input image")
    ap.add_argument(
        "--point",
        action="append",
        default=[],
        help="normalized prompt point 'x,y[,label]' (label 1=fg default, 0=bg); repeatable",
    )
    ap.add_argument("--box", help="normalized prompt box 'x,y,w,h'")
    ap.add_argument("--label", help="free-text hint, best-effort (see README)")
    ap.add_argument("--backend", default="sam2", choices=["sam2", "stub"])
    ap.add_argument("--checkpoint", help="SAM2 checkpoint path (sam2 backend)")
    ap.add_argument("--model-cfg", help="SAM2 model config name/path (sam2 backend)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-nodes", type=int, default=10)
    ap.add_argument("--max-nodes", type=int, default=30)
    args = ap.parse_args()

    points = [_parse_point(p) for p in args.point]
    box = _parse_box(args.box) if args.box else None

    out = segment(
        args.image,
        points=points,
        box=box,
        label=args.label,
        backend=args.backend,
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
        target_min=args.min_nodes,
        target_max=args.max_nodes,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
