"""Main MCP server for darktable integration."""

import base64
import hmac
import json
import logging
import os
import tempfile
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import ImageContent, TextContent, Tool

from .bridge.client import (
    Bridge,
    BridgeError,
    BridgePluginNotInstalledError,
    BridgeTimeoutError,
)
from .darktable.cli_wrapper import CLIWrapper
from .tools.camera_tools import CameraTools
from .tools.matting_tools import run_matting
from .tools.preview_tools import (
    apply_ratings_batch,
    extract_previews,
    format_extract_summary,
    format_open_summary,
    format_ratings_summary,
    open_in_darktable,
)
from .tools.contact_sheet_tools import (
    DIRECTION_VALUES,
    FILTER_VALUES,
    SORT_VALUES,
    compose_sheet,
    effective_thumb_width,
    filter_items,
    new_sheet_path,
    paginate,
    render_thumbnails,
    sort_items,
    validate_limit,
    validate_offset,
    write_sheet,
)
from .tools import retouch_overlay as overlay
from .tools.lut_tools import (
    LutRootNotConfiguredError,
    compose_lut_compare_grid,
    effective_cell_width,
    parse_cube_header,
    resolve_lut_path,
    scan_lut_directory,
)
from .tools.segmentation_tools import run_segmentation
from .utils.errors import DarktableMCPError, MattingServiceError, SegmentationServiceError
from .utils.viewport_coords import (
    POINT_SPACES,
    ViewportCoordinateError,
    canonical_space,
    viewport_point_to_image,
    viewport_radius_to_image,
)

logger = logging.getLogger(__name__)

ToolHandler = Callable[[Dict[str, Any]], Awaitable[List[TextContent]]]


def _remap_bridge_path(container_path: str) -> str:
    """Remap a path returned by the Lua bridge (dev_preview) to one this
    process can actually read.

    On a real workstation install, this server and darktable run on the
    same host, so a bridge path is already directly readable — no remap
    needed. In the containerized dev/test harness
    (docker/run-dt-bridge.sh), darktable runs inside a container with the
    run_dir bind-mounted at /run, so dev_preview reports paths like
    "/run/cache-mcp/darktable-mcp/spike-previews/x.png" that only resolve
    inside the container. Set DARKTABLE_MCP_RUN_DIR to that run_dir (the
    directory docker/run-dt-bridge.sh printed on `start`) to have "/run"
    swapped for it here.
    """
    run_dir = os.environ.get("DARKTABLE_MCP_RUN_DIR")
    if run_dir and container_path and container_path.startswith("/run"):
        return run_dir.rstrip("/") + container_path[len("/run"):]
    return container_path


def _host_to_bridge_path(host_path: str) -> str:
    """Inverse of _remap_bridge_path: a HOST-side path this process just
    wrote (a matte file, T3.3) -> the path the Lua bridge/darktable can
    actually read.

    On a real workstation install both sides share one filesystem, so this
    is the identity. In the docker bridge harness (DARKTABLE_MCP_RUN_DIR
    set), the run_dir tree is bind-mounted into the container at /run, so a
    host path under run_dir maps to the same-named path under /run.
    """
    run_dir = os.environ.get("DARKTABLE_MCP_RUN_DIR")
    if run_dir:
        run_dir = run_dir.rstrip("/")
        if host_path.startswith(run_dir + "/") or host_path == run_dir:
            return "/run" + host_path[len(run_dir):]
    return host_path


def _matte_output_dir() -> Path:
    """Host-writable directory for mask_raster's per-call matte files (T3.3).

    Mirrors the Lua bridge's own cache_dir() convention
    ($XDG_CACHE_HOME/darktable-mcp, falling back to ~/.cache/darktable-mcp)
    so a real workstation install needs no extra configuration. In the
    docker bridge harness, DARKTABLE_MCP_RUN_DIR points at the bind-mounted
    run_dir, and the SAME directory darktable's XDG_CACHE_HOME already uses
    inside the container (see docker/run-dt-bridge.sh: XDG_CACHE_HOME=
    /run/cache-mcp) is reused here on the host side so no new bind mount is
    needed -- $RUN_DIR/cache-mcp/darktable-mcp/mattes is writable by this
    process and visible to the container at /run/cache-mcp/darktable-mcp/
    mattes via _host_to_bridge_path.
    """
    run_dir = os.environ.get("DARKTABLE_MCP_RUN_DIR")
    if run_dir:
        out = Path(run_dir) / "cache-mcp" / "darktable-mcp" / "mattes"
    else:
        cache_home = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        out = Path(cache_home) / "darktable-mcp" / "mattes"
    out.mkdir(parents=True, exist_ok=True)
    return out


# Extensions matte.py's PIL-based loader can read directly. Anything else
# (RAW formats -- CR2/NEF/ARW/RW2/...) needs a rendered substitute first,
# see _handle_mask_raster's "(a) matting input" step.
_DIRECT_MATTE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

# dt_develop_mask_mode_t bit (src-dt/src/develop/blend.h) that must be set
# for a module's DRAWN mask group to actually apply during pixel processing
# -- dt_dev_pixelpipe's _piece_wants_blending gates the blend call on
# DEVELOP_MASK_ENABLED, and blend.c's own mode_drawn check needs this bit.
# Used to verify attach_mask actually turned the mask on (bugreport
# 2026-07-27: wiring the group reference alone left it inert until the user
# clicked the pencil icon by hand), not just trust the write succeeded.
_DEVELOP_MASK_MASK = 1 << 1


def _inline_image_content(
    host_path: str, max_dim: int = 1024, quality: int = 90
) -> Optional[ImageContent]:
    """Downscale + re-encode a rendered frame into a SMALL base64 JPEG that
    reliably fits a remote client's inline-image budget.

    A bare filesystem path is useless to a REMOTE client (ChatGPT over HTTP) --
    it can't read local disk. But a full 1024px preview PNG is ~1MB of base64,
    which ChatGPT silently DROPS from the tool result -- that is the root cause
    of the "preview sometimes shows, sometimes only path/url" bug. Re-encoding
    to JPEG q90 at <= max_dim keeps it ~100-250KB, small enough to always come
    through, so delivery is deterministic. Returns None if the file can't be
    read/decoded (caller falls back to text).
    """
    try:
        import cv2  # bundled (opencv-python-headless), see pyproject deps

        img = cv2.imread(host_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > max_dim:
            scale = max_dim / float(longest)
            img = cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            return None
        return ImageContent(
            type="image",
            data=base64.b64encode(buf.tobytes()).decode("ascii"),
            mimeType="image/jpeg",
        )
    except Exception:  # noqa: BLE001 - never let inlining break the tool result
        return None


def _polygon_bbox(polygon: List[Dict[str, float]]) -> Dict[str, float]:
    """Min/max bbox of a normalized {x,y} point list -- used to report the
    same bbox shape the segmentation sidecar already returns, but for the
    mask-frame polygon actually written to add_path_mask, so a caller can
    compare the two frames directly (see _backtransform_polygon_to_mask_space)."""
    xs = [p["x"] for p in polygon]
    ys = [p["y"] for p in polygon]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def _mask_overlay_content(
    base_png_path: str,
    polygon: List[Dict[str, Any]],
    max_dim: int = 1024,
) -> Optional[ImageContent]:
    """Draw the segmentation polygon over the preview it was computed on, so the
    model can SEE exactly which region got selected.

    This is the fix for auto-masking picking the wrong area (e.g. a bright field
    instead of a small person): the agent looks at the tinted overlay, sees the
    mask is wrong, and re-prompts with a tighter point/box -- instead of blindly
    trusting a bad mask. `polygon` is in pixel coords of `base_png_path` (the
    sidecar segments that exact render), so it overlays 1:1. Returns None on any
    failure.
    """
    try:
        import cv2
        import numpy as np

        img = cv2.imread(base_png_path, cv2.IMREAD_COLOR)
        if img is None or len(polygon) < 3:
            return None
        pts = np.array(
            [[int(p["x"]), int(p["y"])] for p in polygon], dtype=np.int32
        ).reshape((-1, 1, 2))
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))  # BGR red fill
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)  # 40% tint
        cv2.polylines(img, [pts], True, (0, 255, 255), 2)  # yellow outline
        h, w = img.shape[:2]
        longest = max(h, w)
        if longest > max_dim:
            scale = max_dim / float(longest)
            img = cv2.resize(
                img,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            return None
        return ImageContent(
            type="image",
            data=base64.b64encode(buf.tobytes()).decode("ascii"),
            mimeType="image/jpeg",
        )
    except Exception:  # noqa: BLE001
        return None


class _BearerAuthMiddleware:
    """Pure-ASGI static Bearer-token gate for the Streamable HTTP transport.

    Rejects any HTTP request whose Authorization header is not exactly
    ``Bearer <token>`` with a 401. Comparison is constant-time. Kept as a raw
    ASGI wrapper (not Starlette BaseHTTPMiddleware) so it never buffers or
    interferes with the streamed/SSE response body.

    ``bypass_prefix``, when set, exempts paths under it (the ``/mcp/files/``
    download route) from the header check: a remote client like ChatGPT
    fetches that URL itself and cannot attach a custom Authorization header,
    but each download URL already carries its own unguessable per-file token
    (see ``_register_download``), which alone is sufficient bearer capability
    for that one file.
    """

    def __init__(self, app: Any, token: str, bypass_prefix: Optional[str] = None) -> None:
        self._app = app
        self._expected = f"Bearer {token}".encode()
        self._bypass_prefix = bypass_prefix

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        if self._bypass_prefix and scope.get("path", "").startswith(self._bypass_prefix):
            await self._app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        provided = headers.get(b"authorization")
        if provided is None or not hmac.compare_digest(provided, self._expected):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [
                        (b"content-type", b"text/plain; charset=utf-8"),
                        (b"www-authenticate", b"Bearer"),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": b"Unauthorized\n"})
            return
        await self._app(scope, receive, send)


class DarktableMCPServer:
    """MCP server for darktable photo management and editing."""

    def __init__(self) -> None:
        self.app: Server = Server("darktable-mcp")
        self._cli: Optional[CLIWrapper] = None
        self.camera_tools = CameraTools()
        self.bridge = Bridge()
        # Absolute file path of whichever image open_image_in_darkroom last
        # opened successfully, from the bridge's open_darkroom "path" field
        # (T3.3): mask_raster needs the actual image file to feed the
        # matting sidecar and takes no image argument of its own, matching
        # every other darkroom tool's "acts on whichever image is open"
        # convention.
        self._current_image_path: Optional[str] = None
        # Download registry for the HTTP transport: random token -> absolute
        # file path the server produced (previews, exports). The /mcp/files/
        # <token> route streams the mapped file, so a remote client can pull a
        # 120MB TIFF that could never fit in a base64 tool result -- and because
        # a token only ever maps to a path WE registered, there is no way to
        # traverse to an arbitrary file. Capped + FIFO-evicted to bound memory.
        self._download_registry: "OrderedDict[str, str]" = OrderedDict()
        self._download_registry_cap = 1024
        # Viewport snapshot registry (capture_viewport): random token ->
        # {viewport, region, render{path,width,height}, created_at}. This is
        # a BEST-EFFORT snapshot, not an atomic one -- region and render come
        # from two sequential bridge calls, so it can only be wrong if the
        # user pans/zoom mid-call, which the design review (2026-07-25 spec)
        # accepted as out of scope for v1. TTL-expired so a stale snapshot
        # (captured long before an add/update call) is rejected rather than
        # silently retouching wherever the viewport happens to be now.
        self._viewport_snapshots: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._viewport_snapshot_cap = 256
        self._viewport_snapshot_ttl_s = 300.0
        self._handler_map: Dict[str, ToolHandler] = self._build_handlers()
        self._setup_tools()

    @property
    def cli(self) -> CLIWrapper:
        """Get CLI wrapper instance (lazy-loaded)."""
        if self._cli is None:
            self._cli = CLIWrapper()
        return self._cli

    def _setup_tools(self) -> None:
        @self.app.list_tools()
        async def list_tools() -> List[Tool]:
            return self._tool_definitions()

        @self.app.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            # The mcp library's own log line ("Processing request of type
            # CallToolRequest") never names the tool, making the server's
            # stdout useless for "what is actually being called right now"
            # (2026-07-27 feedback). Log it ourselves, one line per call.
            logger.info("tool call: %s(%s)", name, arguments)
            handler = self._handler_map.get(name)
            if handler is None:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
            try:
                return await handler(arguments)
            except DarktableMCPError as e:
                logger.error("Tool %s failed: %s", name, e)
                return [TextContent(type="text", text=f"Error: {e}")]
            except Exception as e:
                logger.exception("Tool %s crashed", name)
                return [TextContent(type="text", text=f"Tool {name} crashed: {e}")]

    def _tool_definitions(self) -> List[Tool]:
        return [
            Tool(
                name="view_photos",
                description=(
                    "Browse photos in the currently open lighttable view (whatever "
                    "collection/filter/filmroll the user has open right now) — not "
                    "the whole library. Filter by filename substring, minimum star "
                    "rating, or both. Returns id, filename, absolute file path, and "
                    "rating per match — the path can be passed straight into "
                    "export_images. Requires darktable to be running with the "
                    "darktable-mcp Lua plugin installed (see darktable-mcp "
                    "install-plugin)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "string",
                            "description": "Substring filter on filename (case-insensitive)",
                        },
                        "rating_min": {
                            "type": "integer",
                            "minimum": -1,
                            "maximum": 5,
                            "description": "Minimum star rating to include",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                            "default": 100,
                            "description": "Maximum number of photos to return",
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["collection", "library"],
                            "default": "collection",
                            "description": (
                                "'collection' (default): only photos in the "
                                "currently open lighttable view. 'library': the "
                                "entire darktable library, ignoring any open "
                                "collection/filter."
                            ),
                        },
                    },
                },
            ),
            Tool(
                name="get_contact_sheet",
                description=(
                    "Build one grid image (a 'contact sheet') from photos in the "
                    "currently open lighttable collection, so a batch can be "
                    "visually culled without opening each photo in darkroom. "
                    "Typical flow: request pages of 25 with filter='unrated', "
                    "look at each sheet, note which image_ids/positions to "
                    "rate or inspect further, then page through with "
                    "next_offset until has_more is false. Read-only -- never "
                    "changes ratings, tags, edit history, or the darkroom "
                    "state. Requires darktable to be running with the "
                    "darktable-mcp Lua plugin installed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                            "description": "Photos to skip, applied AFTER filter+sort",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 64,
                            "default": 25,
                            "description": "Max photos on this sheet (25 is the sweet spot for visual culling)",
                        },
                        "columns": {
                            "type": "integer",
                            "minimum": 2,
                            "maximum": 8,
                            "default": 5,
                            "description": "Grid columns; rows = ceil(returned / columns)",
                        },
                        "filter": {
                            "type": "string",
                            "enum": list(FILTER_VALUES),
                            "default": "unrated",
                            "description": (
                                "'unrated': no stars, not rejected. 'rated': 1-5 stars. "
                                "'rejected': rating -1. 'selected': darktable's current "
                                "lighttable selection. 'all': no filter."
                            ),
                        },
                        "sort": {
                            "type": "string",
                            "enum": list(SORT_VALUES),
                            "default": "filename",
                            "description": "Sort key applied before offset/limit",
                        },
                        "direction": {
                            "type": "string",
                            "enum": list(DIRECTION_VALUES),
                            "default": "asc",
                        },
                        "include_filename": {"type": "boolean", "default": True},
                        "include_image_id": {"type": "boolean", "default": True},
                        "include_rating": {"type": "boolean", "default": True},
                        "include_sequence_number": {"type": "boolean", "default": True},
                        "thumbnail_width": {
                            "type": "integer",
                            "minimum": 200,
                            "maximum": 500,
                            "default": 320,
                            "description": "Per-photo thumbnail width in px; height follows aspect ratio",
                        },
                        "background": {
                            "type": "string",
                            "enum": ["dark", "light"],
                            "default": "dark",
                        },
                    },
                },
            ),
            Tool(
                name="rate_photos",
                description=(
                    "Apply a star rating to one or more photos in the user's "
                    "darktable library. Requires darktable to be running with "
                    "the darktable-mcp Lua plugin installed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "photo_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of photo IDs (from view_photos)",
                        },
                        "rating": {
                            "type": "integer",
                            "minimum": -1,
                            "maximum": 5,
                            "description": "Star rating: -1=reject, 0=unrated, 1-5=stars",
                        },
                    },
                    "required": ["photo_ids", "rating"],
                },
            ),
            Tool(
                name="tag_photo",
                description=(
                    "Attach and/or detach keyword tags on one or more photos "
                    "in the user's darktable library. Tags not already present "
                    "are created automatically. Requires darktable to be "
                    "running with the darktable-mcp Lua plugin installed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "photo_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of photo IDs (from view_photos)",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tag names to attach (created if missing)",
                        },
                        "remove_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tag names to detach",
                        },
                    },
                    "required": ["photo_ids"],
                },
            ),
            Tool(
                name="set_photo_note",
                description=(
                    "Write a description or editorial note for a photo, stored "
                    "in the photo's Description metadata field (Xmp.dc.description "
                    "in the xmp sidecar). Overwrites any existing description on "
                    "that photo. Requires darktable to be running with the "
                    "darktable-mcp Lua plugin installed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "photo_id": {
                            "type": "string",
                            "description": "Photo ID (from view_photos)",
                        },
                        "note": {
                            "type": "string",
                            "description": "Assessment/note text to store as the photo's description",
                        },
                    },
                    "required": ["photo_id", "note"],
                },
            ),
            Tool(
                name="get_photo_note",
                description=(
                    "Read back the Description metadata field (Xmp.dc.description) "
                    "for a photo, e.g. a previously written AI assessment/note. "
                    "Requires darktable to be running with the darktable-mcp Lua "
                    "plugin installed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "photo_id": {
                            "type": "string",
                            "description": "Photo ID (from view_photos)",
                        },
                    },
                    "required": ["photo_id"],
                },
            ),
            Tool(
                name="list_collections",
                description=(
                    "List the user's darktable tags, the closest equivalent to "
                    "Lightroom-style collections since darktable's Lua API has "
                    "no separate collection object. Returns tag name and photo "
                    "count per tag. Requires darktable to be running with the "
                    "darktable-mcp Lua plugin installed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filter": {
                            "type": "string",
                            "description": "Substring filter on tag name (case-insensitive)",
                        },
                    },
                },
            ),
            Tool(
                name="list_photos_in_collection",
                description=(
                    "List photos carrying a given darktable tag (see "
                    "list_collections). Returns id, filename, absolute file "
                    "path, and rating per photo. Requires darktable to be "
                    "running with the darktable-mcp Lua plugin installed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": "Tag/collection name (from list_collections)",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5000,
                            "default": 1000,
                            "description": "Maximum number of photos to return",
                        },
                    },
                    "required": ["collection"],
                },
            ),
            Tool(
                name="import_batch",
                description=(
                    "Register a folder as a film roll in the user's darktable "
                    "library. Useful when you've copied photos from a card or "
                    "external drive and want darktable to know about them. "
                    "Returns the count of newly-imported photos. Requires "
                    "darktable to be running with the darktable-mcp Lua plugin "
                    "installed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_path": {
                            "type": "string",
                            "description": "Absolute path to the folder of photos to import",
                        },
                        "recursive": {
                            "type": "boolean",
                            "default": True,
                            "description": "Recurse into subdirectories (default true)",
                        },
                    },
                    "required": ["source_path"],
                },
            ),
            Tool(
                name="list_styles",
                description=(
                    "List all darktable styles (presets) installed on the user's "
                    "system. Returns name and description for each. Required "
                    "discovery step before calling apply_preset, since style names "
                    "must match exactly. Requires darktable to be running with the "
                    "darktable-mcp Lua plugin installed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="apply_preset",
                description=(
                    "Apply a darktable style (preset) to one or more photos. The "
                    "preset_name must exactly match a style name from list_styles. "
                    "Returns counts of applied and missed photos. Requires "
                    "darktable to be running with the darktable-mcp Lua plugin "
                    "installed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "photo_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Photo IDs (from view_photos)",
                        },
                        "preset_name": {
                            "type": "string",
                            "description": "Style name (must match exactly; see list_styles)",
                        },
                    },
                    "required": ["photo_ids", "preset_name"],
                },
            ),
            Tool(
                name="import_from_camera",
                description=(
                    "Use when a camera or memory card is physically connected. "
                    "Detects the camera via libgphoto2 and copies all photos "
                    "to a local directory. Returns the destination path so the "
                    "user can open darktable and choose 'import folder' to "
                    "register the photos in their library."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "destination": {
                            "type": "string",
                            "description": (
                                "Target directory for copied files. "
                                "Default: ~/Pictures/import-YYYY-MM-DD/"
                            ),
                        },
                        "camera_port": {
                            "type": "string",
                            "description": (
                                "gphoto2 port string (e.g. 'usb:002,002'). "
                                "Required when multiple cameras are connected."
                            ),
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "minimum": 60,
                            "description": (
                                "Subprocess timeout for the transfer. "
                                "Default: 3600 (1 hour). On timeout, re-run "
                                "the tool to resume — already-copied files "
                                "are skipped."
                            ),
                        },
                    },
                },
            ),
            Tool(
                name="extract_previews",
                description=(
                    "Extract auto-rotated JPEG previews and thumbnails from a "
                    "directory of raw files (NEF/CR2/ARW/DNG/etc) for review, "
                    "contact sheets, culling, or downstream image analysis. "
                    "Each preview is rotated upright via EXIF orientation and "
                    "resized to max_dim (default 1024). A smaller thumb_dim "
                    "(default 384) is also written as a lightweight "
                    "first-pass thumbnail. Returns a list of items with "
                    "preview paths plus an EXIF summary (ISO, shutter, "
                    "focal, aperture, datetime) per file."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_dir": {
                            "type": "string",
                            "description": "Directory containing raw files",
                        },
                        "output_dir": {
                            "type": "string",
                            "description": (
                                "Where to write JPEGs. "
                                "Default: <source_dir>/.previews/"
                            ),
                        },
                        "max_dim": {
                            "type": "integer",
                            "minimum": 256,
                            "maximum": 4096,
                            "default": 1024,
                            "description": "Longest-edge for the standard preview",
                        },
                        "thumb_dim": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 1024,
                            "default": 384,
                            "description": "Thumb longest-edge; 0 to skip",
                        },
                        "overwrite": {
                            "type": "boolean",
                            "default": False,
                            "description": "Re-extract even if preview exists",
                        },
                    },
                    "required": ["source_dir"],
                },
            ),
            Tool(
                name="apply_ratings_batch",
                description=(
                    "Write XMP sidecars (xmp:Rating) for a batch of "
                    "{stem: rating} pairs. Sidecars sit next to the raw "
                    "files at <source_dir>/<stem>.<RAW_EXT>.xmp and are "
                    "picked up automatically by darktable on import. "
                    "Rating range: -1 (reject), 0 (unrated), 1-5 (stars). "
                    "Each rating is also appended to "
                    "<source_dir>/ratings.jsonl for replay/audit."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_dir": {
                            "type": "string",
                            "description": "Directory holding the raw files",
                        },
                        "ratings": {
                            "type": "object",
                            "description": (
                                "Map of file stem (e.g. 'DSC_1234') to "
                                "rating int in [-1, 5]"
                            ),
                            "additionalProperties": {
                                "type": "integer",
                                "minimum": -1,
                                "maximum": 5,
                            },
                        },
                        "log": {
                            "type": "boolean",
                            "default": True,
                            "description": "Append entries to ratings.jsonl",
                        },
                    },
                    "required": ["source_dir", "ratings"],
                },
            ),
            Tool(
                name="open_in_darktable",
                description=(
                    "Launch the darktable GUI on a folder. The folder is "
                    "registered as a film roll on first launch and XMP "
                    "sidecars are picked up automatically. The lighttable "
                    "opens already filtered via the official "
                    "`darktable.gui.libs.collect.filter` Lua API for any "
                    "rating spec: exact `rating=N`, `rating_min=N` (>=), "
                    "`rating_max=N` (<=), arbitrary `rating_min..rating_max` "
                    "inner ranges, or no filter at all."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_dir": {
                            "type": "string",
                            "description": "Folder containing the raw files",
                        },
                        "rating": {
                            "type": "integer",
                            "minimum": -1,
                            "maximum": 5,
                            "description": (
                                "Filter to exactly this rating "
                                "(-1=reject, 0=unrated, 1-5=stars)"
                            ),
                        },
                        "rating_min": {
                            "type": "integer",
                            "minimum": -1,
                            "maximum": 5,
                            "description": "Lower bound of a rating range",
                        },
                        "rating_max": {
                            "type": "integer",
                            "minimum": -1,
                            "maximum": 5,
                            "description": "Upper bound of a rating range",
                        },
                        "darktable_path": {
                            "type": "string",
                            "default": "darktable",
                            "description": "darktable executable (default: 'darktable' on PATH)",
                        },
                    },
                    "required": ["source_dir"],
                },
            ),
            Tool(
                name="export_images",
                description=(
                    "Export photos to JPEG/PNG/TIFF via darktable-cli. "
                    "Pass absolute file paths in photo_ids — the `path` field "
                    "from view_photos drops in directly. IMPORTANT if any "
                    "photo has multiple duplicates/versions (darktable's "
                    "own duplicate manager): without an explicit xmp_paths "
                    "entry, darktable-cli auto-detects `<path>.xmp` -- "
                    "always the BASE/version-0 duplicate's sidecar, "
                    "silently ignoring any other version, even the one "
                    "currently open in darkroom (bugreport 2026-07-31: "
                    "exported an older edit instead of the one being "
                    "worked on). Pass the `sidecar` field from view_photos "
                    "or get_current_image in xmp_paths (same order/length "
                    "as photo_ids, use null for entries that should use "
                    "the default) whenever you're exporting a SPECIFIC "
                    "edited version rather than a fresh, never-duplicated "
                    "image."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "photo_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Absolute paths to source images",
                        },
                        "xmp_paths": {
                            "type": "array",
                            "items": {"type": ["string", "null"]},
                            "description": (
                                "Optional, same length/order as photo_ids: "
                                "explicit .xmp sidecar path per file (from "
                                "view_photos's/get_current_image's "
                                "`sidecar` field) -- see the tool "
                                "description for why this matters on a "
                                "duplicated/versioned image. null (or a "
                                "shorter list) falls back to darktable-"
                                "cli's own auto-detected sidecar for that "
                                "entry."
                            ),
                        },
                        "output_path": {"type": "string"},
                        "format": {
                            "type": "string",
                            "enum": ["jpeg", "png", "tiff"],
                        },
                        "quality": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                        },
                    },
                    "required": ["photo_ids", "output_path", "format"],
                },
            ),
            # ---- Phase 1 (T1.6): scalar darkroom editing loop --------------
            # These five tools drive the conversational discuss -> edit ->
            # preview -> iterate loop over the darktable.develop.* Lua API
            # (see PLAN.md §3). Typical sequence for one editing step:
            #   list_modules() -> get_params(op) -> set_params(op, fields)
            #   -> get_preview() -> look at the image -> repeat.
            # Scope: scalar/bool/enum fields (exposure, temperature, tint,
            # contrast, saturation, ...) AND whole-array field writes (T1.7:
            # colorbalancergb, channelmixerrgb rows, tonecurve nodes). Masks are
            # handled by the dedicated add_path_mask/mask_object/mask_raster
            # tools, not get_params/set_params.
            Tool(
                name="open_image_in_darkroom",
                description=(
                    "Open a photo in the darktable darkroom so its edit can be "
                    "inspected and changed. Call this FIRST, before "
                    "list_modules/get_params/set_params/get_preview, which all "
                    "act on whichever image is currently open. image_id comes "
                    "from view_photos. Requires darktable to be running with "
                    "the darktable-mcp Lua bridge loaded."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "image_id": {
                            "type": "integer",
                            "description": "Image id (from view_photos) to open in darkroom",
                        },
                    },
                    "required": ["image_id"],
                },
            ),
            Tool(
                name="navigate_photo",
                description=(
                    "Move to the next or previous photo in the currently open "
                    "lighttable collection and open it in darkroom, replacing "
                    "whatever image is open now. Requires an image already open "
                    "in darkroom (open_image_in_darkroom or opened by hand) so "
                    "there's a reference point to move from. Ordering matches "
                    "get_contact_sheet's sort/direction (default: filename asc) "
                    "-- pass the same sort/direction_order to stay consistent "
                    "with the sheet you're paging through. Errors clearly at "
                    "either end of the collection instead of wrapping around."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "direction": {
                            "type": "string",
                            "enum": ["next", "previous"],
                            "description": "Which neighbor to open",
                        },
                        "sort": {
                            "type": "string",
                            "enum": list(SORT_VALUES),
                            "default": "filename",
                            "description": "Ordering to compute 'neighbor' from",
                        },
                        "direction_order": {
                            "type": "string",
                            "enum": list(DIRECTION_VALUES),
                            "default": "asc",
                            "description": "asc/desc for the sort key above",
                        },
                    },
                    "required": ["direction"],
                },
            ),
            Tool(
                name="get_current_image",
                description=(
                    "Report which image is currently open in the darktable "
                    "darkroom: {has_image, id, path, filename}. Works whether "
                    "the image was opened via open_image_in_darkroom OR opened "
                    "BY HAND in the GUI. Use it to confirm the edit target "
                    "before other darkroom tools, or to recover the image id/"
                    "path when you did not open the image yourself. "
                    "has_image=false means no image is open in darkroom yet."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="list_modules",
                description=(
                    "List the processing modules active on the image currently "
                    "open in darkroom (exposure, white balance/temperature, "
                    "colorbalancergb, tonecurve, ...), each as "
                    "{op, instance, enabled}. Call this first to discover which "
                    "op names exist on this image before calling get_params/"
                    "set_params — op must match one of these exactly. instance "
                    "is 0 for the base copy of a module and increments for "
                    "extra masked instances (add_instance, Phase 2)."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="get_params",
                description=(
                    "Read the current field values of one module instance on "
                    "the open image, e.g. get_params('exposure') returns "
                    "{'exposure': {'value':0.0,'min':-3.0,'max':3.0,"
                    "'default':0.0}, 'black': {...}, ...}. Scalar/bool/enum "
                    "fields come back with value/min/max/default; array-valued "
                    "fields (e.g. colorbalancergb, channelmixerrgb mixing rows, "
                    "tonecurve nodes) come back as arrays. Always call this "
                    "before set_params so a proposed new value can be grounded "
                    "in the real min/max/default rather than guessed — e.g. "
                    "before proposing '+0.35 EV' for exposure, check its max is "
                    "at least that far above the current value. Note: get_params "
                    "does not surface per-index min/max for array fields."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Module operation name, e.g. 'exposure', 'temperature', 'colorbalancergb' (see list_modules)",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the module instance; 0 = base instance",
                        },
                    },
                    "required": ["op"],
                },
            ),
            Tool(
                name="set_params",
                description=(
                    "Write one or more fields on a module instance of the open "
                    "image (e.g. set_params('exposure', {'exposure': -1.5})), "
                    "commit them to darkroom history, and reprocess. Scalar, "
                    "bool and enum fields are supported, AND whole-array field "
                    "writes work too — e.g. set_params('channelmixerrgb', "
                    "{'grey': [0.65, 0.28, 0.07, 0]}) sets a float[4] mixing "
                    "row, and tonecurve/colorbalancergb array fields can be "
                    "written wholesale. Caveat: arrays are replaced whole (no "
                    "partial single-index write). Requested values are ALWAYS "
                    "CLAMPED to that "
                    "field's min/max from get_params — they are never "
                    "rejected. The returned report has 'applied' (the values "
                    "actually written, post-clamp), 'clamped' (a list of "
                    "{field, requested, applied, min, max} for every field "
                    "that hit a bound — if the user's ask exceeded a bound, "
                    "tell them, e.g. 'that's already at max, +5 EV clamped to "
                    "+3 EV'), and 'unknown_fields' (names that don't exist on "
                    "this module, ignored rather than fatal). fields may also "
                    "include a convenience 'enabled': true/false — it is "
                    "routed to enable_module BEFORE the rest of fields is "
                    "applied and never appears in unknown_fields, so a module "
                    "that's off by default (grain, sharpen, vignette, "
                    "tonecurve, ...) can be enabled and configured in one "
                    "call, e.g. set_params('sharpen', {'enabled': True, "
                    "'amount': 2.0}). After calling this, call get_preview() "
                    "to see the visual result before deciding on a further "
                    "nudge."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Module operation name, e.g. 'exposure' (see list_modules)",
                        },
                        "fields": {
                            "type": "object",
                            "description": (
                                "Map of field name to new value, e.g. "
                                "{'exposure': -1.5}. Field names come from "
                                "get_params(op). An optional 'enabled' bool "
                                "is a convenience routed to enable_module "
                                "instead of the native set_params call."
                            ),
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the module instance; 0 = base instance",
                        },
                    },
                    "required": ["op", "fields"],
                },
            ),
            Tool(
                name="get_blend_params",
                description=(
                    "Read a module's BLEND state (opacity, blend mode, mask "
                    "mode) -- a SEPARATE flat struct from the introspected "
                    "fields get_params returns, so it needs its own tool. "
                    "This is how you check/adjust a module's overall "
                    "strength when the module itself has no 'amount' field "
                    "of its own -- e.g. lut3d, which is otherwise all-or-"
                    "nothing at 100%. opacity is 0..100 (percent, matches "
                    "the GUI's own blend-opacity slider). mask_mode is a "
                    "bitmask (0=off, 1=uniformly/opacity-only, higher bits = "
                    "drawn/parametric/raster masks already set up by other "
                    "tools -- treat it as informational here, see "
                    "set_blend_params for the one safe write path)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Module operation name, e.g. 'lut3d' (see list_modules)",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the module instance; 0 = base instance",
                        },
                    },
                    "required": ["op"],
                },
            ),
            Tool(
                name="set_blend_params",
                description=(
                    "Set a module's overall blend opacity (0..100 percent, "
                    "e.g. 30 for a subtle LUT/effect, 100 for full strength) "
                    "-- the fix for modules like lut3d that have no 'amount' "
                    "of their own and are otherwise all-or-nothing. Pass "
                    "enable_uniform_blend=true the FIRST time you set opacity "
                    "on a module (mask_mode defaults to fully off, so opacity "
                    "alone is a silent no-op until this is set once); it "
                    "OVERWRITES mask_mode wholesale to plain uniform blending "
                    "-- do NOT use this on a module you've masked via "
                    "mask_object/mask_raster/retouch/set_raster_source, since "
                    "it would clobber that mask wiring. Those tools manage "
                    "their own blend_params bits directly; this one is for "
                    "the simple 'run this module at X% strength, no mask' case."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Module operation name, e.g. 'lut3d' (see list_modules)",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the module instance; 0 = base instance",
                        },
                        "opacity": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Blend opacity, 0..100 percent; clamped, never rejected",
                        },
                        "enable_uniform_blend": {
                            "type": "boolean",
                            "description": (
                                "true: set mask_mode to plain uniform blending "
                                "(opacity applies, no mask). false: disable "
                                "blending entirely (module runs at native 100%, "
                                "opacity ignored). Omit to leave mask_mode "
                                "untouched -- only do this if you already know "
                                "it's set up correctly."
                            ),
                        },
                        "blend_mode": {
                            "type": "integer",
                            "description": "Advanced: raw dt_develop_blend_mode_t value; omit unless you specifically need a non-normal blend mode",
                        },
                    },
                    "required": ["op"],
                },
            ),
            Tool(
                name="list_masks",
                description=(
                    "List EVERY drawn mask shape in the current image, "
                    "regardless of which module(s) currently use it -- so you "
                    "can find a shape drawn earlier (by hand in the GUI, or "
                    "via add_path_mask/mask_object/retouch_add_shape earlier "
                    "this session) and wire it into a NEW module instance "
                    "with attach_mask, instead of guessing its formid or "
                    "redrawing it. Each entry reports `used_by` (which "
                    "op/instance pairs already reference this shape) -- a "
                    "shape can legally back several modules at once."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="get_module_mask",
                description=(
                    "Read module (op, instance)'s full mask state in one "
                    "call: the group-level opacity/mask_mode/blend_mode/"
                    "invert (from get_blend_params) PLUS the list of drawn "
                    "shapes wired into its blend group, each with its own "
                    "boolean-combine operation (union/intersection/"
                    "difference/exclusion) and per-shape invert. Use this "
                    "before attach_mask/detach_mask to see what is already "
                    "there, and after them to confirm the change landed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Module operation name, e.g. 'exposure' (see list_modules)",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the module instance; 0 = base instance",
                        },
                    },
                    "required": ["op"],
                },
            ),
            Tool(
                name="get_mask_geometry",
                description=(
                    "Read one drawn mask's full point geometry: for a "
                    "path/brush shape, every node's corner/ctrl1/ctrl2/"
                    "border(feather) coordinates plus a per-node state "
                    "code; for circle/ellipse, its single center/radius "
                    "descriptor. Use this to verify a mask's actual shape "
                    "after creation (e.g. whether a node is a smooth or a "
                    "corner point: ctrl1/ctrl2 equal to corner means "
                    "corner, differing means smooth -- there is no single "
                    "'smooth' flag on the mask as a whole, it's per node) "
                    "instead of re-segmenting to check. Points are in the "
                    "same PIPE-INPUT/mask-frame convention add_path_mask "
                    "expects on write -- NOT the display frame get_preview "
                    "renders (see mask_object's bbox_display_frame/"
                    "bbox_mask_frame for that distinction)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "mask_id": {
                            "type": "integer",
                            "description": (
                                "The shape's own formid -- from list_masks/"
                                "list_all_masks, or the 'formid' returned "
                                "by add_path_mask/mask_object/"
                                "retouch_add_shape."
                            ),
                        },
                    },
                    "required": ["mask_id"],
                },
            ),
            Tool(
                name="rename_mask",
                description=(
                    "Give a drawn mask a caller-chosen name instead of "
                    "the auto-generated 'path #7'/'circle #3' -- useful "
                    "once a session has created several masks and formid "
                    "alone is hard to keep straight (e.g. name one "
                    "'model body', another 'background sky'). Purely "
                    "cosmetic -- does not touch geometry, opacity, or "
                    "which module(s) reference the shape."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "mask_id": {
                            "type": "integer",
                            "description": "The shape's own formid (see get_mask_geometry)",
                        },
                        "name": {
                            "type": "string",
                            "description": "New display name for the mask",
                        },
                    },
                    "required": ["mask_id", "name"],
                },
            ),
            Tool(
                name="delete_mask",
                description=(
                    "Permanently delete a drawn mask shape, whether or "
                    "not it's currently attached to any module -- unlike "
                    "detach_mask, which only unwires a shape from ONE "
                    "module's blend group and leaves the shape itself "
                    "around forever. Use this to clean up an orphan/"
                    "unwanted mask (e.g. left over from calibration or a "
                    "failed attempt) that list_all_masks shows with an "
                    "empty or stale `used_by`. Every module still "
                    "referencing this shape loses that reference "
                    "(mirrors darktable's own mask-manager delete action)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "mask_id": {
                            "type": "integer",
                            "description": "The shape's own formid (see list_all_masks/get_mask_geometry)",
                        },
                    },
                    "required": ["mask_id"],
                },
            ),
            Tool(
                name="attach_mask",
                description=(
                    "Wire an EXISTING drawn mask shape (formid, from "
                    "list_masks or the return value of add_path_mask/"
                    "mask_object/retouch_add_shape) into module (op, "
                    "instance)'s blend group WITHOUT copying it -- the same "
                    "shape stays a single underlying object, now referenced "
                    "by this module IN ADDITION to whatever already used it. "
                    "Typical use: draw shapes once by hand in the GUI (or "
                    "via add_path_mask), create several new module instances "
                    "(add_instance), then attach the right shape(s) to each "
                    "instance's mask instead of redrawing per instance. Fails "
                    "loudly if formid does not exist or the module does not "
                    "support blending -- never silently no-ops. Also turns "
                    "the module's drawn-mask blend bit ON (equivalent to "
                    "clicking the mask pencil icon in the GUI) and verifies "
                    "it stuck by reading blend params back -- otherwise the "
                    "shape would be wired in but invisible/inactive until "
                    "someone toggled it by hand."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Module operation name, e.g. 'exposure' (see list_modules)",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the module instance; 0 = base instance",
                        },
                        "formid": {
                            "type": "integer",
                            "description": "Shape id to attach, from list_masks() or a mask-creating tool's return value",
                        },
                        "operation": {
                            "type": "string",
                            "enum": ["union", "intersection", "difference", "exclusion"],
                            "default": "union",
                            "description": (
                                "How this shape combines with whatever else "
                                "is already in the module's mask group. "
                                "Irrelevant for the first shape attached to "
                                "an empty group."
                            ),
                        },
                    },
                    "required": ["op", "formid"],
                },
            ),
            Tool(
                name="detach_mask",
                description=(
                    "Unwire a drawn mask shape from module (op, instance)'s "
                    "blend group WITHOUT deleting the shape itself -- it "
                    "stays available in list_masks() and can be re-attached "
                    "(to this module or a different one) via attach_mask. If "
                    "this was the LAST shape in the group, the module's mask "
                    "is fully cleared (mirrors the GUI's own 'no masks' "
                    "action) -- expected, not an error."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Module operation name, e.g. 'exposure' (see list_modules)",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the module instance; 0 = base instance",
                        },
                        "formid": {
                            "type": "integer",
                            "description": "Shape id to detach, from get_module_mask/list_masks",
                        },
                    },
                    "required": ["op", "formid"],
                },
            ),
            Tool(
                name="set_module_mask",
                description=(
                    "Replace module (op, instance)'s ENTIRE set of attached "
                    "shapes in one call: shapes not in the new list are "
                    "detached (not deleted -- still available for other "
                    "modules), shapes in the new list that are not yet "
                    "attached are attached, and shapes already attached with "
                    "a DIFFERENT operation are re-attached with the new one. "
                    "Optionally also sets the group's overall opacity/invert "
                    "in the same call (forwarded to set_blend_params). Prefer "
                    "attach_mask/detach_mask for a single incremental change; "
                    "use this when you know the FULL desired shape list "
                    "up front (e.g. scripting several new instances from the "
                    "same set of hand-drawn shapes)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Module operation name, e.g. 'exposure' (see list_modules)",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the module instance; 0 = base instance",
                        },
                        "shapes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "formid": {"type": "integer"},
                                    "operation": {
                                        "type": "string",
                                        "enum": ["union", "intersection", "difference", "exclusion"],
                                        "default": "union",
                                    },
                                },
                                "required": ["formid"],
                            },
                            "description": "The complete desired shape list (formid + combine operation each). Empty list clears the module's mask.",
                        },
                        "opacity": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100,
                            "description": "Optional: also set overall blend opacity (0..100), forwarded to set_blend_params",
                        },
                        "invert": {
                            "type": "boolean",
                            "description": "Optional: also set the whole-group mask polarity, forwarded to set_blend_params",
                        },
                    },
                    "required": ["op", "shapes"],
                },
            ),
            Tool(
                name="enable_module",
                description=(
                    "Turn a module instance on or off and commit to darkroom "
                    "history. Many modules ship OFF by default (grain, "
                    "sharpen, vignette, tonecurve, ...) and produce NO visible "
                    "effect until enabled, even after set_params — call this "
                    "before (or together, via set_params's 'enabled' "
                    "convenience field) writing params on those. Also useful "
                    "to A/B a module's effect by toggling it and comparing "
                    "get_preview() output."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Module operation name, e.g. 'grain', 'sharpen' (see list_modules)",
                        },
                        "enabled": {
                            "type": "boolean",
                            "description": "True to enable, false to disable",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the module instance; 0 = base instance",
                        },
                    },
                    "required": ["op", "enabled"],
                },
            ),
            Tool(
                name="list_luts",
                description=(
                    "List LUT files (.cube, .3dl, .png haldclut) under the "
                    "lut3d module's configured root directory -- the SAME "
                    "folder darktable's own file-chooser dropdown reads "
                    "(plugins/darkroom/lut3d/def_path), so this listing "
                    "matches what a human sees in the UI. Each result: "
                    "{name, path, format, category}; for .cube files, "
                    "'title' and 'size' (LUT_3D_SIZE, e.g. 17/33/65) are "
                    "included when the file's header declares them. 'path' "
                    "is relative to the root and is exactly what set_params's "
                    "lut3d 'filepath' field expects -- copy it straight "
                    "through, no further resolution needed. 'category' is "
                    "the LUT's first-level subfolder (e.g. 'FG'), '' if it "
                    "sits directly in the root. Scans recursively regardless "
                    "of 'directory' (the UI's own combobox only shows one "
                    "folder at a time; this surfaces the whole tree, or a "
                    "scoped subtree when 'directory' is given). Fails with a "
                    "clear message if no LUT folder has ever been configured "
                    "in darktable."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "directory": {
                            "type": "string",
                            "description": (
                                "Optional subdirectory under the LUT root to "
                                "scope the scan to, e.g. 'FG'. Omit to scan "
                                "the whole root."
                            ),
                        },
                    },
                },
            ),
            Tool(
                name="preview_lut",
                description=(
                    "Try a LUT (path from list_luts) on the open image WITHOUT "
                    "a lasting edit: apply it to lut3d, render a preview, then "
                    "restore lut3d to exactly the state it was in before this "
                    "call (filepath, colorspace, interpolation, enabled/disabled) "
                    "-- so trying 20 LUTs in a row leaves darkroom history "
                    "unchanged. Note darktable already coalesces consecutive "
                    "edits of the SAME module into one history entry, so even "
                    "set_params directly would not spam history; this tool's "
                    "real job is the automatic restore, not history hygiene. "
                    "Fails validation up front if the resolved file does not "
                    "exist under the LUT root (see list_luts). colorspace/"
                    "interpolation are optional strings -- check "
                    "get_params('lut3d').fields.colorspace.options / "
                    ".interpolation.options for the valid enum labels; omit "
                    "either to keep whatever lut3d is currently set to."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "LUT path relative to the configured root (list_luts' 'path' field)",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the lut3d module instance; 0 = base instance",
                        },
                        "colorspace": {
                            "type": "string",
                            "description": "Optional enum label, e.g. 'DT_IOP_SRGB'; omit to keep current",
                        },
                        "interpolation": {
                            "type": "string",
                            "description": "Optional enum label, e.g. 'DT_IOP_TETRAHEDRAL'; omit to keep current",
                        },
                        "opacity": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": (
                                "Optional blend opacity 0..1 (lut3d has no "
                                "'amount' of its own -- e.g. 0.3 for a subtle "
                                "look). Restored afterward along with everything "
                                "else. Refused if this instance already has a "
                                "mask configured (see set_blend_params) -- omit "
                                "to run the LUT at its current blend state."
                            ),
                        },
                        "max_w": {"type": "integer", "default": 1024},
                        "max_h": {"type": "integer", "default": 1024},
                        "region": {
                            "type": "object",
                            "description": "Optional {x,y,w,h} normalized 0..1 sub-rectangle, same convention as get_preview",
                        },
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="compare_luts",
                description=(
                    "Render the SAME open image once per LUT in `paths` (each "
                    "via preview_lut -- applied, rendered, restored, one at a "
                    "time) and stitch the results into one labelled grid image "
                    "so several looks can be judged side by side in a single "
                    "glance instead of N separate preview_lut calls. NOT "
                    "atomic across paths: each LUT is independently applied "
                    "and restored, so one bad path (missing file, invalid "
                    "enum) shows as an ERROR cell labelled with its path "
                    "rather than aborting the rest. colorspace/interpolation, "
                    "if given, apply to every LUT in the comparison."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": "LUT paths relative to the configured root (list_luts' 'path' field)",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the lut3d module instance; 0 = base instance",
                        },
                        "colorspace": {
                            "type": "string",
                            "description": "Optional enum label applied to every LUT in the comparison",
                        },
                        "interpolation": {
                            "type": "string",
                            "description": "Optional enum label applied to every LUT in the comparison",
                        },
                        "opacity": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Optional blend opacity 0..1 applied to every LUT in the comparison, same as preview_lut's",
                        },
                        "columns": {"type": "integer", "default": 4, "minimum": 1},
                        "cell_width": {
                            "type": "integer",
                            "default": 480,
                            "description": "Per-cell render width in pixels (shrunk if the grid would exceed the max sheet width)",
                        },
                        "background": {
                            "type": "string",
                            "enum": ["dark", "light"],
                            "default": "dark",
                        },
                    },
                    "required": ["paths"],
                },
            ),
            Tool(
                name="add_instance",
                description=(
                    "Add a new masked/parametric instance of a module "
                    "(mirrors the GUI 'new instance' action) and return its "
                    "instance index. This is the base step for local edits — "
                    "e.g. a second 'exposure' instance for local dodge/burn, "
                    "or a second 'sharpen' instance masked to one area — "
                    "since instance 0 stays the global edit and the new "
                    "instance can be masked and adjusted independently via "
                    "get_params/set_params with instance=<returned index>. "
                    "Call list_modules() afterward to see both instances.\n\n"
                    "IMPORTANT for local edits: a new instance is created by "
                    "COPYING instance 0's params verbatim, not by recomputing "
                    "this module's own defaults for a fresh instance — for "
                    "some modules that copy is wrong for a LOCAL edit. "
                    "exposure is the known case: instance 0 typically has "
                    "compensate_exposure_bias/compensate_hilite_pres on "
                    "(global exposure compensation), but a local dodge/burn "
                    "instance almost always wants both OFF so its exposure "
                    "value is a pure, unadjusted offset. Pass `fields` to set "
                    "the new instance's initial params in the SAME call/"
                    "history entry, instead of a separate set_params call "
                    "after the fact."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Module operation name, e.g. 'exposure' (see list_modules)",
                        },
                        "fields": {
                            "type": "object",
                            "description": (
                                "Optional initial param overrides for the new "
                                "instance, applied in the same history entry "
                                "as its creation. Same field names/semantics "
                                "as set_params(op). For a local exposure "
                                "instance: {'compensate_exposure_bias': "
                                "false, 'compensate_hilite_pres': false}."
                            ),
                        },
                    },
                    "required": ["op"],
                },
            ),
            Tool(
                name="get_viewport",
                description=(
                    "Read the darkroom canvas zoom/pan state for the main "
                    "window and, if open on a second monitor, the preview2 "
                    "window ({'main': {...}, 'preview2': {'active': bool, "
                    "...}}). Use this to know what part/zoom of the image "
                    "the user is currently looking at. Each viewport carries a "
                    "ready-to-use 'region' {x,y,w,h} (top-left, normalized 0..1, "
                    "clamped to [0,1]) = the crop of the full image visible in "
                    "that window: pass it STRAIGHT INTO get_preview's 'region' "
                    "argument to render exactly that crop at full detail. Do NOT "
                    "use the raw 'zoom_x'/'zoom_y' fields for that — they are "
                    "center-relative and can be negative, so they are not valid "
                    "as a top-left region. 'region' is absent when the viewport "
                    "has no processed pipe yet or preview2 is inactive (treat as "
                    "full-frame)."
                ),
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="get_preview",
                description=(
                    "Render a PNG of the CURRENT LIVE darkroom edit (the "
                    "in-memory pixelpipe result after your set_params calls) "
                    "and return its file path so you can view it. This does "
                    "NOT write to the database or XMP sidecar — it's a "
                    "cheap, repeatable look at the live edit. Call it after "
                    "every set_params (or a small batch of them) to check the "
                    "visual effect before proposing the next adjustment — "
                    "this is the 'preview' step of the discuss -> edit -> "
                    "preview -> iterate loop. max_w/max_h cap the rendered "
                    "size (aspect ratio preserved, never upscaled beyond the "
                    "darkroom's own preview resolution). Pass 'region' "
                    "({x,y,w,h}, each normalized 0..1 of the visible frame) "
                    "to render just a sub-rectangle at full detail instead of "
                    "the whole downscaled frame — use this to inspect "
                    "grain/sharpening/noise, which are easy to miss at "
                    "full-frame scale; pair it with get_viewport() to match "
                    "the region the user is actually looking at."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "max_w": {
                            "type": "integer",
                            "default": 1024,
                            "minimum": 16,
                            "description": "Maximum preview width in pixels",
                        },
                        "max_h": {
                            "type": "integer",
                            "default": 1024,
                            "minimum": 16,
                            "description": "Maximum preview height in pixels",
                        },
                        "return_image": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "Return the render inline as a viewable image "
                                "(default true). Set false for metadata/path/url "
                                "only when you don't need to see it."
                            ),
                        },
                        "inline_max_dim": {
                            "type": "integer",
                            "default": 1024,
                            "minimum": 64,
                            "description": (
                                "Longest-side cap (px) for the inline image. It "
                                "is downscaled + JPEG-encoded to this so it always "
                                "fits the client's inline-image budget; the "
                                "full-res PNG stays at the returned path/url."
                            ),
                        },
                        "region": {
                            "type": "object",
                            "description": (
                                "Optional sub-rectangle to render at full "
                                "detail: {x,y,w,h}, each normalized 0..1 of "
                                "the visible frame (x,y = top-left corner)."
                            ),
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0, "maximum": 1},
                                "w": {"type": "number", "minimum": 0, "maximum": 1},
                                "h": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                        },
                    },
                },
            ),
            Tool(
                name="capture_viewport",
                description=(
                    "Snapshot the region + a full-detail render of what the "
                    "user is CURRENTLY looking at in 'main' or 'preview2' "
                    "(the external/second-monitor darkroom window), so you "
                    "can pick retouch points directly off this render's own "
                    "pixels or 0..1 frame instead of the full image. Returns "
                    "a snapshot_id -- pass it to retouch_add_shape_in_viewport "
                    "/ retouch_update_shape_in_viewport so the point you pick "
                    "and the render you looked at are guaranteed to agree. "
                    "The snapshot is BOUND to the image open in darkroom at "
                    "capture time: a later add/update call refuses to write if "
                    "darktable has moved to a different image, instead of "
                    "retouching the wrong photo. Pixel coordinates you pass "
                    "later are pixels of THIS render (the reported render WxH), "
                    "not of the darktable window. "
                    "The snapshot is BEST-EFFORT, not atomic (region and "
                    "render come from two sequential reads): fine as long as "
                    "the user isn't actively panning/zooming mid-call. "
                    "Snapshots expire after a few minutes -- re-capture if a "
                    "later add/update call reports snapshot_not_found_or_expired. "
                    "Errors with viewport_not_active if 'preview2' is requested "
                    "but the second window isn't open."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "viewport": {
                            "type": "string",
                            "enum": ["main", "preview2"],
                            "default": "main",
                            "description": "Which darkroom window to capture",
                        },
                        "max_w": {
                            "type": "integer",
                            "default": 1400,
                            "minimum": 16,
                            "description": "Maximum render width in pixels",
                        },
                        "max_h": {
                            "type": "integer",
                            "default": 1400,
                            "minimum": 16,
                            "description": "Maximum render height in pixels",
                        },
                        "return_image": {
                            "type": "boolean",
                            "default": True,
                            "description": "Return the render inline (downscaled JPEG) as well as its path",
                        },
                    },
                },
            ),
            Tool(
                name="set_viewport",
                description=(
                    "Write the darkroom zoom/pan for 'main' or 'preview2' -- "
                    "the counterpart to get_viewport()/capture_viewport, whose "
                    "render detail is otherwise capped by whatever zoom the "
                    "human last happened to leave it at. Give exactly ONE of "
                    "'region' (a sub-rectangle of the full PROCESSED, post-crop "
                    "frame, normalized 0..1, top-left origin -- the SAME frame "
                    "get_viewport().region / get_preview(region=...) use, no "
                    "third coordinate frame), 'scale' (1.0 = 100%, 1 image px : "
                    "1 screen px; keeps the current pan), or 'mode' (fit/fill/"
                    "100%/200%). If the requested region's aspect does not "
                    "match the window, the region is EXPANDED (never cropped) "
                    "about its own center so the whole request stays visible -- "
                    "see 'aspect_adjusted'/'achieved.region' in the response. "
                    "By default (wait_for_pipe=true) this blocks until the "
                    "pipe has actually reprocessed at the new zoom before "
                    "returning, so an immediately-following capture_viewport "
                    "never returns a stale, pre-zoom render -- if the bounded "
                    "wait times out, 'pipe_ready' is false rather than a false "
                    "'ok'. 'scale' is clamped to darktable's own zoom range; "
                    "check 'clamped' for whether that happened. Errors with "
                    "viewport_not_active if 'preview2' is requested but the "
                    "second window is not open (no state change). IMPORTANT: "
                    "capture_viewport's own pixel source (dev->preview_pipe) "
                    "has a FIXED native resolution independent of this call's "
                    "zoom -- 'renderable_px' in the response is a real probe "
                    "of what capture_viewport will actually return at the "
                    "achieved region, and 'min_render_px_across' is honored "
                    "on a best-effort basis: if the probe shows it can't be "
                    "met, this does NOT zoom in further to chase it (that "
                    "would shrink, not grow, the available pixels here -- see "
                    "'note' in the response when this happens) and it does "
                    "NOT report a false success. Call restore_viewport with "
                    "the returned 'previous' when done, so you don't leave the "
                    "human's darkroom zoomed into a random detail."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "viewport": {
                            "type": "string",
                            "enum": ["main", "preview2"],
                            "default": "main",
                            "description": "Which darkroom window to zoom/pan",
                        },
                        "region": {
                            "type": "object",
                            "description": (
                                "Sub-rectangle of the full processed frame to "
                                "show, normalized 0..1, top-left origin: "
                                "{x,y,w,h}. Mutually exclusive with scale/mode."
                            ),
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0, "maximum": 1},
                                "w": {"type": "number", "minimum": 0, "maximum": 1},
                                "h": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                        },
                        "scale": {
                            "type": "number",
                            "description": (
                                "Absolute zoom scale, 1.0 = 100%. Keeps the "
                                "current pan. Mutually exclusive with region/mode."
                            ),
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["fit", "fill", "100%", "200%"],
                            "description": (
                                "Named zoom preset. Mutually exclusive with "
                                "region/scale."
                            ),
                        },
                        "min_render_px_across": {
                            "type": "integer",
                            "minimum": 1,
                            "description": (
                                "Best-effort hint only -- see the tool "
                                "description's IMPORTANT note. Reports "
                                "min_render_px_across_satisfied instead of "
                                "silently pretending success when it can't be met."
                            ),
                        },
                        "wait_for_pipe": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "Block until the pipe has actually reprocessed "
                                "at the new zoom before returning."
                            ),
                        },
                        "timeout_ms": {
                            "type": "integer",
                            "default": 4000,
                            "minimum": 100,
                            "description": "Bound for the wait_for_pipe wait, in milliseconds",
                        },
                    },
                },
            ),
            Tool(
                name="restore_viewport",
                description=(
                    "Put a viewport's zoom/pan back exactly as it was before "
                    "a set_viewport call -- pass the SAME 'previous' object "
                    "set_viewport (or get_viewport) returned, unmodified. Call "
                    "this when you are done with a set_viewport-driven "
                    "retouch/inspection loop so you don't leave the human's "
                    "darkroom zoomed into a random detail. Same wait_for_pipe/ "
                    "timeout_ms/pipe_ready semantics as set_viewport; no "
                    "scale clamping (a state that was once valid is restored "
                    "as-is). Errors with viewport_not_active if 'preview2' is "
                    "requested but the second window is not open."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "viewport": {
                            "type": "string",
                            "enum": ["main", "preview2"],
                            "default": "main",
                            "description": "Which darkroom window to restore",
                        },
                        "previous": {
                            "type": "object",
                            "description": (
                                "The exact object set_viewport's/get_viewport's "
                                "'previous'/viewport state returned earlier -- "
                                "{zoom, closeup, zoom_x, zoom_y, scale}."
                            ),
                        },
                        "wait_for_pipe": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "Block until the pipe has actually reprocessed "
                                "at the restored zoom before returning."
                            ),
                        },
                        "timeout_ms": {
                            "type": "integer",
                            "default": 4000,
                            "minimum": 100,
                            "description": "Bound for the wait_for_pipe wait, in milliseconds",
                        },
                    },
                    "required": ["previous"],
                },
            ),
            Tool(
                name="retouch_add_shape_in_viewport",
                description=(
                    "Like retouch_add_shape, but target/source/radius/feather "
                    "are given relative to a snapshot from capture_viewport "
                    "instead of the full image -- the server converts them "
                    "for you, so you never hand-compute the full-image "
                    "fraction from a cropped/zoomed render. Rejects (does "
                    "NOT clamp) target or source points that fall outside "
                    "the captured viewport/render, and rejects an "
                    "expired/unknown snapshot_id -- both come back as an "
                    "explicit error so a stale or wrong point never silently "
                    "retouches the wrong spot -- as does a snapshot whose "
                    "image is no longer the one open in darkroom. Set "
                    "return_preview to get a "
                    "render of the SAME region back immediately so you can "
                    "verify the result without a second capture_viewport call."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "snapshot_id": {
                            "type": "string",
                            "description": "snapshot_id from a prior capture_viewport call",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the retouch module instance",
                        },
                        "algorithm": {
                            "type": "string",
                            "enum": ["heal", "clone"],
                            "description": "Retouch algorithm for this shape",
                        },
                        "coordinate_space": {
                            "type": "string",
                            "enum": list(POINT_SPACES),
                            "default": "snapshot_normalized",
                            "description": (
                                "'snapshot_normalized' (alias 'viewport_normalized'): "
                                "0..1 within the captured render. 'snapshot_pixels' "
                                "(alias 'viewport_pixels'): pixel coordinates of the "
                                "RENDER capture_viewport returned (its reported "
                                "render WxH) -- NOT the darktable window's pixel "
                                "size, which is usually larger."
                            ),
                        },
                        "radius_space": {
                            "type": "string",
                            "enum": list(POINT_SPACES),
                            "description": (
                                "Space for 'radius'/'feather'. Defaults to "
                                "coordinate_space. IMPORTANT: a snapshot_pixels "
                                "radius is scaled by the render's WIDTH only "
                                "(never height, never an average) so circle "
                                "size doesn't depend on the window's aspect ratio."
                            ),
                        },
                        "target": {
                            "type": "object",
                            "description": "Shape center, in coordinate_space units",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                            "required": ["x", "y"],
                        },
                        "source": {
                            "type": "object",
                            "description": "Source point to sample from (absolute, not an offset), in coordinate_space units",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                            "required": ["x", "y"],
                        },
                        "radius": {
                            "type": "number",
                            "minimum": 0,
                            "description": "Circle radius, in radius_space units",
                        },
                        "feather": {
                            "type": "number",
                            "default": 0.0,
                            "minimum": 0,
                            "description": "Soft edge width, in radius_space units",
                        },
                        "opacity": {
                            "type": "number",
                            "default": 1.0,
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Shape opacity (1.0 = full effect)",
                        },
                        "wavelet_scale": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Wavelet scale to retouch on; defaults to the module's current scale",
                        },
                        "return_preview": {
                            "type": "boolean",
                            "default": True,
                            "description": "Render the same captured region again after the edit, for immediate visual verification",
                        },
                    },
                    "required": ["snapshot_id", "algorithm", "target", "source", "radius"],
                },
            ),
            Tool(
                name="retouch_update_shape_in_viewport",
                description=(
                    "Move/resize an EXISTING retouch shape (by formid, from "
                    "retouch_add_shape_in_viewport's response or "
                    "retouch_list_shapes) in place -- same coordinate "
                    "conversion and containment rejection as "
                    "retouch_add_shape_in_viewport, but keeps the shape's "
                    "identity instead of deleting and recreating it. "
                    "target/source/radius/feather must be resent in full "
                    "each call (this moves the whole shape, not a single field)."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "snapshot_id": {
                            "type": "string",
                            "description": "snapshot_id from a prior capture_viewport call",
                        },
                        "formid": {
                            "type": "integer",
                            "description": "Shape form id to move/resize",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the retouch module instance",
                        },
                        "algorithm": {
                            "type": "string",
                            "enum": ["heal", "clone"],
                            "description": "Optional: change the shape's algorithm; omit to keep it unchanged",
                        },
                        "coordinate_space": {
                            "type": "string",
                            "enum": list(POINT_SPACES),
                            "default": "snapshot_normalized",
                            "description": (
                                "Same convention as retouch_add_shape_in_viewport: "
                                "snapshot_pixels are pixels of the capture_viewport "
                                "render, not of the darktable window"
                            ),
                        },
                        "radius_space": {
                            "type": "string",
                            "enum": list(POINT_SPACES),
                            "description": "Same convention as retouch_add_shape_in_viewport; defaults to coordinate_space",
                        },
                        "target": {
                            "type": "object",
                            "description": "New shape center, in coordinate_space units",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                            "required": ["x", "y"],
                        },
                        "source": {
                            "type": "object",
                            "description": "New source point, in coordinate_space units",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                            "required": ["x", "y"],
                        },
                        "radius": {
                            "type": "number",
                            "minimum": 0,
                            "description": "New circle radius, in radius_space units",
                        },
                        "feather": {
                            "type": "number",
                            "default": 0.0,
                            "minimum": 0,
                            "description": "New soft edge width, in radius_space units",
                        },
                        "opacity": {
                            "type": "number",
                            "description": "Optional: change the shape's opacity; omit to leave it unchanged",
                        },
                        "wavelet_scale": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Optional: change the wavelet scale; omit to keep it unchanged",
                        },
                        "return_preview": {
                            "type": "boolean",
                            "default": True,
                            "description": "Render the same captured region again after the edit, for immediate visual verification",
                        },
                    },
                    "required": ["snapshot_id", "formid", "target", "source", "radius"],
                },
            ),
            Tool(
                name="retouch_delete_shapes",
                description=(
                    "Delete multiple retouch shapes by formid in one call -- "
                    "batch version of retouch_delete_shape, so cleaning up "
                    "several wrong points doesn't need one approval per "
                    "shape. NOT atomic: each formid is deleted independently, "
                    "so check the 'failed' list in the response."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the retouch module instance",
                        },
                        "formids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                            "description": "Shape form ids to remove",
                        },
                    },
                    "required": ["formids"],
                },
            ),
            Tool(
                name="retouch_render_overlay",
                description=(
                    "Draw the retouch shapes on top of a capture_viewport "
                    "snapshot so you can SEE where they landed: target circle, "
                    "feather ring, source circle and the source->target link, "
                    "labelled by formid. This is how you verify a heal point "
                    "actually covers the mark, is not oversized, its feather "
                    "does not spill onto an edge, and its source sits on clean "
                    "texture -- checks that plain numbers cannot answer. "
                    "The overlay is drawn by this server from the shape "
                    "geometry, NOT captured from darktable's own on-screen "
                    "overlay (that one is painted on the GUI widget and is "
                    "absent from every render we can read), so it needs no "
                    "focus, changes no darktable state, and additionally "
                    "reports overlapping shapes. modes: all_shapes (every "
                    "shape; highlight_formid optionally dims the rest), "
                    "selected_shape (one shape prominent, others dimmed for "
                    "context), source_and_target (only the chosen shape, "
                    "nothing else), mask_only (the actual mask alpha as a "
                    "greyscale image, same falloff darktable applies -- use it "
                    "to judge coverage, not composition). Read-only: refuses "
                    "if darkroom has moved to another image, since the "
                    "geometry would then belong to a different photo."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "snapshot_id": {
                            "type": "string",
                            "description": "Snapshot id from capture_viewport (the render to draw on)",
                        },
                        "mode": {
                            "type": "string",
                            "enum": list(overlay.MODES),
                            "default": overlay.MODE_ALL_SHAPES,
                            "description": "What to draw (see the tool description)",
                        },
                        "highlight_formid": {
                            "type": "integer",
                            "description": (
                                "Shape to emphasise. Required by selected_shape "
                                "and source_and_target unless the module has "
                                "exactly one shape; optional for all_shapes"
                            ),
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the retouch module instance",
                        },
                        "label_shapes": {
                            "type": "boolean",
                            "default": True,
                            "description": "Print each shape's formid next to its circle",
                        },
                        "return_image": {
                            "type": "boolean",
                            "default": True,
                            "description": "Return the overlay inline (downscaled JPEG) as well as its path",
                        },
                    },
                    "required": ["snapshot_id"],
                },
            ),
            # ---- Phase 2 (T2.3): object masks -------------------------------
            Tool(
                name="add_path_mask",
                description=(
                    "You almost certainly want mask_object instead (does "
                    "vision-pick -> segment -> mask -> edit in one call) -- "
                    "LOW-LEVEL: attach a drawn polygon path mask to a "
                    "module instance so its effect only applies inside "
                    "that shape. Use add_path_mask directly only if you "
                    "already have a normalized polygon from somewhere "
                    "(e.g. you already called mask_object and want to "
                    "attach the same polygon to a different module/"
                    "instance). points must be an array of >=3 {x,y} "
                    "nodes normalized 0..1 -- but NOT against get_preview's "
                    "frame. This is the MASK STORAGE frame: pipe-input, "
                    "pre-crop, and rotated per EXIF orientation -- NOT the "
                    "post-crop/post-rotate frame get_preview/"
                    "capture_viewport render. The two are identical only "
                    "when no crop/rotate/orientation is active. Hand-"
                    "computing this frame from a preview image (e.g. "
                    "deriving it empirically by placing a mask, exporting, "
                    "and looking where it landed) is the #1 cause of "
                    "misplaced shapes -- mask_object does this conversion "
                    "for you internally; prefer it whenever the prompt is "
                    "point/box/label-shaped instead of an already-known "
                    "polygon."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Module operation name, e.g. 'exposure' (see list_modules)",
                        },
                        "points": {
                            "type": "array",
                            "minItems": 3,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number", "minimum": 0, "maximum": 1},
                                    "y": {"type": "number", "minimum": 0, "maximum": 1},
                                },
                                "required": ["x", "y"],
                            },
                            "description": "Polygon nodes, normalized 0..1, boundary order",
                        },
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the module instance to mask",
                        },
                        "opacity": {
                            "type": "number",
                            "default": 1.0,
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Mask opacity (1.0 = full effect inside the shape)",
                        },
                        "feather": {
                            "type": "number",
                            "default": 0.02,
                            "minimum": 0.0,
                            "maximum": 0.5,
                            "description": (
                                "Edge softening as a FRACTION of the mask's own "
                                "bounding box (0.02 = 2%), so a small subject "
                                "gets a proportionally small soft edge instead "
                                "of a frame-constant halo."
                            ),
                        },
                        "smooth": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "Smooth the boundary with Catmull-Rom bezier "
                                "curves so rounded subjects are not faceted. "
                                "Set false for a straight-segment polygon."
                            ),
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Optional display name for the new mask "
                                "(default is darktable's own auto-generated "
                                "'path #7' etc) -- useful for orientation "
                                "once a session has created several masks "
                                "on one image. Equivalent to calling "
                                "rename_mask right after this returns."
                            ),
                        },
                    },
                    "required": ["op", "points"],
                },
            ),
            Tool(
                name="retouch_add_shape",
                description=(
                    "You almost certainly want "
                    "retouch_add_shape_in_viewport instead -- it takes "
                    "coordinates in the SAME frame capture_viewport/"
                    "get_preview render in and converts them for you. "
                    "This tool's target/source/radius are normalized 0..1 "
                    "against the MASK STORAGE frame -- pipe-input, "
                    "pre-crop, rotated per EXIF orientation -- NOT the "
                    "post-crop/post-rotate frame get_preview renders. "
                    "Hand-deriving this frame from a preview image is the "
                    "#1 cause of misplaced shapes. "
                    "Create a local HEAL or CLONE circle shape on the retouch "
                    "module — the module's actual local-editing surface, "
                    "distinct from add_path_mask's generic 'restrict this "
                    "module's blend to a region'. Fixes sensor dust, small "
                    "surface marks, or localized image artifacts by sampling "
                    "a source region onto a target region, optionally on a "
                    "specific wavelet scale (fine detail vs base tones vs "
                    "residual). Only "
                    "'circle' shapes and 'heal'/'clone' algorithms are "
                    "supported so far — ellipse/path/brush and blur/fill are "
                    "a later phase. Requires a retouch module instance to "
                    "already exist in the open image's history (call "
                    "enable_module/add_instance first if needed) and "
                    "darktable running with the darktable-mcp Lua plugin. "
                    "Call get_preview() afterwards to see the result."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the retouch module instance",
                        },
                        "algorithm": {
                            "type": "string",
                            "enum": ["heal", "clone"],
                            "description": "Retouch algorithm for this shape",
                        },
                        "shape_type": {
                            "type": "string",
                            "enum": ["circle"],
                            "default": "circle",
                            "description": "Only 'circle' is supported so far",
                        },
                        "target": {
                            "type": "object",
                            "description": "Shape center, normalized 0..1 against the MASK STORAGE frame (pipe-input, pre-crop) -- NOT get_preview's frame. Use retouch_add_shape_in_viewport instead if unsure.",
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["x", "y"],
                        },
                        "source": {
                            "type": "object",
                            "description": (
                                "Source point to sample from, normalized "
                                "0..1. Required for heal/clone — this is an "
                                "ABSOLUTE point, not an offset from target."
                            ),
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["x", "y"],
                        },
                        "radius": {
                            "type": "number",
                            "minimum": 0.0005,
                            "maximum": 0.5,
                            "description": "Circle radius, normalized 0..1 against the mask storage frame's mindim(width,height) -- see target's description.",
                        },
                        "feather": {
                            "type": "number",
                            "default": 0.0,
                            "minimum": 0.0,
                            "maximum": 0.5,
                            "description": "Soft edge width, normalized 0..1",
                        },
                        "opacity": {
                            "type": "number",
                            "default": 1.0,
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Shape opacity (1.0 = full effect)",
                        },
                        "wavelet_scale": {
                            "type": "integer",
                            "minimum": 0,
                            "description": (
                                "Wavelet scale to retouch on: 0 = original "
                                "image, 1..N = a specific detail scale, N+1 "
                                "= residual. Defaults to the module's "
                                "current scale if omitted — check "
                                "retouch_list_shapes for num_scales/curr_scale."
                            ),
                        },
                    },
                    "required": ["algorithm", "target", "source", "radius"],
                },
            ),
            Tool(
                name="retouch_delete_shape",
                description=(
                    "Delete a retouch shape by its formid (from "
                    "retouch_add_shape or retouch_list_shapes). Requires "
                    "darktable running with the darktable-mcp Lua plugin."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the retouch module instance",
                        },
                        "formid": {
                            "type": "integer",
                            "description": "Shape form id to remove",
                        },
                    },
                    "required": ["formid"],
                },
            ),
            Tool(
                name="retouch_list_shapes",
                description=(
                    "List the real, currently-attached retouch shapes for a "
                    "module instance — unlike get_params(op='retouch'), "
                    "which exposes the raw 300-slot internal array full of "
                    "mostly-empty padding, this returns only actual shapes "
                    "plus the module's wavelet-scale state (num_scales, "
                    "curr_scale, merge_from_scale). Requires darktable "
                    "running with the darktable-mcp Lua plugin."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "instance": {
                            "type": "integer",
                            "default": 0,
                            "description": "multi_priority of the retouch module instance",
                        },
                    },
                },
            ),
            Tool(
                name="mask_object",
                description=(
                    "Apply a local edit to ONE object/region of the open "
                    "image, picked visually — e.g. brighten a foreground "
                    "object, darken the sky, adjust a garment. YOU (the "
                    "vision model) look at a "
                    "get_preview image, pick point(s) and/or a box on the "
                    "subject, and pass them here together with the desired "
                    "adjustment. This tool then: (1) re-renders the full "
                    "preview frame so coordinates line up exactly with your "
                    "picks, (2) segments your points/box into a precise "
                    "object polygon, (3) creates a new masked module "
                    "instance, (4) attaches the polygon as a path mask so "
                    "the effect ONLY applies inside the object, (5) applies "
                    "your adjustment on that instance, and (6) renders a "
                    "fresh preview so you can confirm the result. "
                    "Segmentation works OUT OF THE BOX with no extra "
                    "install: it uses a rough OpenCV GrabCut segmenter "
                    "bundled in this server. Install the optional SAM2 "
                    "sidecar (see dist/INSTALL-sidecar.md, or run "
                    "'darktable-mcp install-sidecar') for precise, "
                    "SAM2-quality masks — when the sidecar is configured "
                    "and reachable it is tried first and used instead. The "
                    "result reports which backend was actually used "
                    "('sam2' or 'grabcut') so you know the quality tier. "
                    "Coordinates for points/box are normalized 0..1 against "
                    "the FULL image frame (same convention as get_preview's "
                    "own 'region' and get_viewport's 'region' — if you're "
                    "looking at a get_preview(region=...) crop, remember "
                    "its pixels map back into that sub-rectangle, not the "
                    "full 0..1 range). Give at least one of points or box. "
                    "On any failure (both segmentation backends down, "
                    "mask/param write error) nothing is left behind — no "
                    "orphan module instance."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": (
                                "Module to apply the local edit with, e.g. "
                                "'exposure' for a brightness lift, "
                                "'colorbalancergb' for a color/tone shift "
                                "(see list_modules for what's available)"
                            ),
                        },
                        "adjustment": {
                            "type": "object",
                            "description": (
                                "Field map applied via set_params on the new "
                                "masked instance, e.g. {'exposure': 0.6}. "
                                "Same shape/clamping rules as set_params's "
                                "'fields' — check get_params(op) first to "
                                "ground the value in real min/max."
                            ),
                        },
                        "points": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "x": {"type": "number", "minimum": 0, "maximum": 1},
                                    "y": {"type": "number", "minimum": 0, "maximum": 1},
                                    "label": {
                                        "type": "integer",
                                        "enum": [0, 1],
                                        "default": 1,
                                        "description": "1 = include (foreground), 0 = exclude (background)",
                                    },
                                },
                                "required": ["x", "y"],
                            },
                            "description": (
                                "One or more prompt points, normalized 0..1 "
                                "against the full frame. Usually one "
                                "label:1 point on the object center is "
                                "enough for SAM2. If the segmented mask "
                                "bleeds into a nearby distractor (e.g. "
                                "branches behind a subject, a shadow next "
                                "to it), add one or more label:0 points ON "
                                "the distractor to explicitly exclude it -- "
                                "mixing several label:1 (include) and "
                                "label:0 (exclude) points in one call is "
                                "the single cheapest way to fix a wrong-"
                                "shaped mask without changing anything else."
                            ),
                        },
                        "box": {
                            "type": "object",
                            "description": (
                                "A bounding box around the subject instead "
                                "of/in addition to points, normalized 0..1: "
                                "{x,y,w,h}, x/y = top-left corner."
                            ),
                            "properties": {
                                "x": {"type": "number", "minimum": 0, "maximum": 1},
                                "y": {"type": "number", "minimum": 0, "maximum": 1},
                                "w": {"type": "number", "minimum": 0, "maximum": 1},
                                "h": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                        },
                        "label": {
                            "type": "string",
                            "description": (
                                "Free-text description of the subject "
                                "(e.g. 'the dog', 'a face', 'the red car'). "
                                "If given ALONE (no points/box), this is "
                                "resolved into a real box via Grounding "
                                "DINO (a text-grounded detector) BEFORE "
                                "segmentation runs -- SAM2 itself has no "
                                "text grounding, but this resolution step "
                                "gives label-only prompts real localization "
                                "instead of doing nothing. If the described "
                                "subject isn't found with enough confidence "
                                "the call fails outright (no guess, no "
                                "silent whole-image mask) -- pick points/box "
                                "yourself instead in that case. If points/"
                                "box are ALSO given, they take priority and "
                                "this becomes best-effort/inert again (old "
                                "behavior, unchanged)."
                            ),
                        },
                        "opacity": {
                            "type": "number",
                            "default": 1.0,
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Mask opacity (1.0 = full effect inside the object)",
                        },
                        "new_instance": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "True (default): create a new masked "
                                "instance of op, leaving instance 0's global "
                                "edit untouched. False: mask+edit instance 0 "
                                "directly (rare — usually wrong for a "
                                "'global' module already in use)."
                            ),
                        },
                        "smooth": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "True (default): Bezier-smooth the path "
                                "between polygon nodes, same as darktable's "
                                "own drawn-path default. False: sharp corner "
                                "nodes, no curve interpolation between them "
                                "-- try this if a segmented mask has visible "
                                "overshoot loops or bulges on a shape with "
                                "many sharp concave corners (Bezier smoothing "
                                "between sparse nodes can overshoot there; "
                                "corner nodes track the polygon exactly, at "
                                "the cost of a slightly more angular edge)."
                            ),
                        },
                        "max_nodes": {
                            "type": "integer",
                            "default": 48,
                            "minimum": 10,
                            "maximum": 200,
                            "description": (
                                "The simplifier converges the polygon's "
                                "node count TOWARD this value (not just a "
                                "ceiling it rarely reaches -- fixed "
                                "2026-07-27, previously raising this had "
                                "no real effect). Raise it (e.g. 80-140) "
                                "for a complex/non-convex silhouette (a "
                                "body in an unusual pose, an object with "
                                "several limbs/protrusions) that's losing "
                                "real shape detail at the default -- more "
                                "nodes track the true contour more "
                                "closely and reduce Bezier overshoot on "
                                "sharp concave corners, at the cost of a "
                                "slightly heavier path for darktable to "
                                "render."
                            ),
                        },
                        "min_nodes": {
                            "type": "integer",
                            "default": 10,
                            "minimum": 3,
                            "maximum": 200,
                            "description": (
                                "Floor on the simplified polygon's node "
                                "count -- rarely needs changing (max_nodes "
                                "is the knob that actually controls "
                                "detail level); only matters if you also "
                                "want to force a MINIMUM density on an "
                                "otherwise very simple/convex shape."
                            ),
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Optional display name for the new mask -- "
                                "NOT the same as `label` (which is a "
                                "segmentation PROMPT, e.g. 'the red car'; "
                                "this is just a human-readable name like "
                                "'model body' for orientation once a "
                                "session has created several masks on one "
                                "image). Default is darktable's own "
                                "auto-generated 'path #7' etc."
                            ),
                        },
                    },
                    "required": ["op", "adjustment"],
                },
            ),
            # ---- Phase 3 (T3.3): raster (matte) masks ----------------------
            Tool(
                name="mask_raster",
                description=(
                    "Apply a local edit gated by a SOFT-EDGED alpha matte "
                    "for subjects with fine or semi-transparent edges "
                    "(hair, fur, fabric fibers, motion blur, glass, smoke) "
                    "instead of a hard drawn polygon -- use this instead of "
                    "mask_object whenever the subject has fine edges a path "
                    "mask would jag up. It acts on whichever image is "
                    "currently open in darkroom (call "
                    "open_image_in_darkroom first) with NO point/box "
                    "picking needed -- unlike mask_object, the MODNet "
                    "matting model generates a dense, whole-image, "
                    "unprompted foreground matte, so there is nothing to "
                    "pick. "
                    "Flow: (1) run the image (or, for RAW files the matting "
                    "model can't read directly, a full-resolution darkroom "
                    "preview export) through the MODNet matting sidecar to "
                    "get a continuous 0..1 alpha, (2) write it to a fresh, "
                    "uniquely-named file and load it into a NEW instance of "
                    "the stock 'rasterfile' module, (3) create a new (or "
                    "reuse instance 0 of) the target module, apply your "
                    "adjustment, (4) wire the target module's blend to "
                    "consume rasterfile's raster mask via set_raster_source, "
                    "(5) render a fresh preview. REQUIRES the matting "
                    "sidecar (MODNet, see dist/INSTALL-sidecar.md) -- there "
                    "is NO lower-quality fallback (a hard-edged segmenter "
                    "cannot produce a soft matte, so this errors clearly "
                    "instead of silently degrading). On ANY failure "
                    "(matting sidecar down, wiring error) nothing is left "
                    "behind -- no orphan rasterfile or consumer module "
                    "instance."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": (
                                "Module to apply the matte-limited edit "
                                "with, e.g. 'colorbalancergb' for a warm/cool "
                                "shift, 'exposure' for a brightness lift "
                                "(see list_modules)"
                            ),
                        },
                        "adjustment": {
                            "type": "object",
                            "description": (
                                "Field map applied via set_params on the "
                                "consumer instance, e.g. {'exposure': 0.6}. "
                                "Check get_params(op) first to ground the "
                                "value in real min/max."
                            ),
                        },
                        "opacity": {
                            "type": "number",
                            "default": 1.0,
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": (
                                "Overall matte strength (1.0 = the matte's "
                                "own alpha values apply at full strength; "
                                "lower values scale the whole effect down "
                                "uniformly, same convention as mask_object's "
                                "opacity)."
                            ),
                        },
                        "new_instance": {
                            "type": "boolean",
                            "default": True,
                            "description": (
                                "True (default): create a new masked "
                                "instance of op, leaving instance 0's global "
                                "edit untouched. False: mask+edit instance 0 "
                                "directly."
                            ),
                        },
                    },
                    "required": ["op", "adjustment"],
                },
            ),
        ]

    def _build_handlers(self) -> Dict[str, ToolHandler]:
        return {
            "import_from_camera": self._handle_import_from_camera,
            "export_images": self._handle_export_images,
            "extract_previews": self._handle_extract_previews,
            "apply_ratings_batch": self._handle_apply_ratings_batch,
            "open_in_darktable": self._handle_open_in_darktable,
            "view_photos": self._handle_view_photos,
            "get_contact_sheet": self._handle_get_contact_sheet,
            "rate_photos": self._handle_rate_photos,
            "tag_photo": self._handle_tag_photo,
            "set_photo_note": self._handle_set_photo_note,
            "get_photo_note": self._handle_get_photo_note,
            "list_collections": self._handle_list_collections,
            "list_photos_in_collection": self._handle_list_photos_in_collection,
            "import_batch": self._handle_import_batch,
            "list_styles": self._handle_list_styles,
            "apply_preset": self._handle_apply_preset,
            "open_image_in_darkroom": self._handle_open_image_in_darkroom,
            "navigate_photo": self._handle_navigate_photo,
            "get_current_image": self._handle_get_current_image,
            "list_modules": self._handle_list_modules,
            "get_params": self._handle_get_params,
            "set_params": self._handle_set_params,
            "get_blend_params": self._handle_get_blend_params,
            "set_blend_params": self._handle_set_blend_params,
            "list_masks": self._handle_list_masks,
            "get_module_mask": self._handle_get_module_mask,
            "get_mask_geometry": self._handle_get_mask_geometry,
            "rename_mask": self._handle_rename_mask,
            "delete_mask": self._handle_delete_mask,
            "attach_mask": self._handle_attach_mask,
            "detach_mask": self._handle_detach_mask,
            "set_module_mask": self._handle_set_module_mask,
            "get_preview": self._handle_get_preview,
            "enable_module": self._handle_enable_module,
            "list_luts": self._handle_list_luts,
            "preview_lut": self._handle_preview_lut,
            "compare_luts": self._handle_compare_luts,
            "add_instance": self._handle_add_instance,
            "get_viewport": self._handle_get_viewport,
            "set_viewport": self._handle_set_viewport,
            "restore_viewport": self._handle_restore_viewport,
            "capture_viewport": self._handle_capture_viewport,
            "add_path_mask": self._handle_add_path_mask,
            "retouch_add_shape": self._handle_retouch_add_shape,
            "retouch_add_shape_in_viewport": self._handle_retouch_add_shape_in_viewport,
            "retouch_update_shape_in_viewport": self._handle_retouch_update_shape_in_viewport,
            "retouch_delete_shape": self._handle_retouch_delete_shape,
            "retouch_delete_shapes": self._handle_retouch_delete_shapes,
            "retouch_list_shapes": self._handle_retouch_list_shapes,
            "retouch_render_overlay": self._handle_retouch_render_overlay,
            "mask_object": self._handle_mask_object,
            "mask_raster": self._handle_mask_raster,
        }

    def list_tools(self) -> List[str]:
        """Tool names registered with the server (used by tests/introspection)."""
        return list(self._handler_map.keys())

    async def _handle_import_from_camera(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.camera_tools.import_from_camera(arguments)
            return [TextContent(type="text", text=result)]
        except Exception as e:
            logger.error("import_from_camera failed: %s", e)
            return [TextContent(type="text", text=f"Error: {e}")]

    async def _handle_export_images(self, arguments: Dict[str, Any]) -> List[TextContent]:
        photo_ids = arguments.get("photo_ids") or []
        output_path = arguments.get("output_path")
        format_type = arguments.get("format", "jpeg")
        quality = int(arguments.get("quality", 95))
        xmp_paths_arg = arguments.get("xmp_paths")

        if not output_path:
            return [TextContent(type="text", text="output_path is required")]
        if not photo_ids:
            return [
                TextContent(
                    type="text",
                    text="photo_ids must contain at least one path",
                )
            ]
        if xmp_paths_arg is not None and len(xmp_paths_arg) != len(photo_ids):
            return [TextContent(
                type="text",
                text=(
                    f"xmp_paths length ({len(xmp_paths_arg)}) must match "
                    f"photo_ids length ({len(photo_ids)}) -- use null for "
                    "entries that should use the default sidecar."
                ),
            )]

        input_files = [Path(p) for p in photo_ids]
        xmp_paths = (
            [Path(p) if p else None for p in xmp_paths_arg] if xmp_paths_arg is not None else None
        )
        out_dir = Path(output_path)
        results = self.cli.batch_export(
            input_files=input_files,
            output_dir=out_dir,
            xmp_paths=xmp_paths,
            format_type=format_type,
            quality=quality,
        )
        # Stash per-file results in a JSONL side file. The full per-file map
        # blew Claude's token budget at 400+ files, so the response stays
        # short and the agent reads the side file when it actually wants
        # details.
        side_file = out_dir / ".export_images.jsonl"
        out_dir.mkdir(parents=True, exist_ok=True)
        ok = fail = 0
        first_error: Optional[str] = None
        ext = format_type.lower()
        download_urls: List[str] = []  # (successful exports only)
        with side_file.open("w") as fh:
            for src, status in results.items():
                is_error = status.startswith("Error") or "failed" in status.lower()
                rec: Dict[str, Any] = {"input": src, "status": status}
                if is_error:
                    fail += 1
                    if first_error is None:
                        first_error = f"{src}: {status[:200]}"
                else:
                    ok += 1
                    # Full-res download URL so a remote client can stream the
                    # exported file (a 120MB TIFF can't ride inline in a result).
                    out_file = out_dir / f"{Path(src).stem}.{ext}"
                    url = self._download_url(str(out_file))
                    if url:
                        rec["url"] = url
                        download_urls.append(url)
                fh.write(json.dumps(rec) + "\n")
        summary = [
            f"exported: {ok}, failed: {fail}",
            f"output_dir: {out_dir}",
            f"details: {side_file} (JSONL, one line per file: input, status, url)",
        ]
        if first_error:
            summary.append(f"first error: {first_error}")
        if download_urls:
            # Cap inline URLs so a 400-file batch doesn't blow the token budget;
            # every URL is still in the JSONL side file above.
            cap = 25
            summary.append(
                f"download URLs (no auth header needed, token in URL is the "
                f"credential) [{min(len(download_urls), cap)} of {len(download_urls)} shown]:"
            )
            summary.extend(download_urls[:cap])
            if len(download_urls) > cap:
                summary.append(f"... {len(download_urls) - cap} more in {side_file}")
        return [TextContent(type="text", text="\n".join(summary))]

    async def _handle_extract_previews(self, arguments: Dict[str, Any]) -> List[TextContent]:
        source_dir = arguments.get("source_dir")
        if not source_dir:
            return [TextContent(type="text", text="source_dir is required")]
        result = extract_previews(
            source_dir=source_dir,
            output_dir=arguments.get("output_dir"),
            max_dim=int(arguments.get("max_dim", 1024)),
            thumb_dim=int(arguments.get("thumb_dim", 384)),
            overwrite=bool(arguments.get("overwrite", False)),
        )
        return [TextContent(type="text", text=format_extract_summary(result))]

    async def _handle_open_in_darktable(self, arguments: Dict[str, Any]) -> List[TextContent]:
        source_dir = arguments.get("source_dir")
        if not source_dir:
            return [TextContent(type="text", text="source_dir is required")]
        result = open_in_darktable(
            source_dir=source_dir,
            rating=arguments.get("rating"),
            rating_min=arguments.get("rating_min"),
            rating_max=arguments.get("rating_max"),
            darktable_path=arguments.get("darktable_path", "darktable"),
        )
        return [TextContent(type="text", text=format_open_summary(result))]

    async def _handle_view_photos(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            photos = self.bridge.call("view_photos", arguments)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if not photos:
            return [TextContent(type="text", text="No photos found matching criteria")]
        # Surface the absolute file path so the agent can hand it straight to
        # export_images (which takes file paths in `photo_ids`). Without this
        # the two tools don't compose: view_photos returns IDs, export wants
        # paths, and the agent has no way to bridge the two.
        lines = [f"Found {len(photos)} photos:"]
        for p in photos:
            stars = "⭐" * (p.get("rating") or 0)
            path = p.get("path") or ""
            lines.append(f"ID: {p['id']} | {p['filename']} | Rating: {stars} | {path}")
        return [TextContent(type="text", text="\n".join(lines))]

    async def _handle_get_contact_sheet(self, arguments: Dict[str, Any]) -> List[Any]:
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", 25)
        columns = int(arguments.get("columns", 5))
        filter_name = arguments.get("filter", "unrated")
        sort_name = arguments.get("sort", "filename")
        direction = arguments.get("direction", "asc")
        thumbnail_width = int(arguments.get("thumbnail_width", 320))
        background = arguments.get("background", "dark")
        include_filename = bool(arguments.get("include_filename", True))
        include_image_id = bool(arguments.get("include_image_id", True))
        include_rating = bool(arguments.get("include_rating", True))
        include_sequence_number = bool(arguments.get("include_sequence_number", True))

        offset_err = validate_offset(offset)
        if offset_err:
            return [TextContent(type="text", text=json.dumps({"status": "error", "error": offset_err}))]
        limit_err = validate_limit(limit)
        if limit_err:
            return [TextContent(type="text", text=json.dumps(
                {"status": "error", "error": limit_err, "message": "limit must be between 1 and 64"}
            ))]
        if filter_name not in FILTER_VALUES:
            return [TextContent(type="text", text=json.dumps(
                {"status": "error", "error": "INVALID_FILTER", "message": f"filter must be one of {FILTER_VALUES}"}
            ))]
        if sort_name not in SORT_VALUES:
            return [TextContent(type="text", text=json.dumps(
                {"status": "error", "error": "INVALID_SORT", "message": f"sort must be one of {SORT_VALUES}"}
            ))]

        try:
            all_images = self.bridge.call("get_collection_images", {"scope": "collection"}, timeout=10.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        matching = filter_items(all_images, filter_name)
        matching = sort_items(matching, sort_name, direction)
        page, next_offset, has_more = paginate(matching, offset, limit)

        base_response = {
            "offset": offset,
            "limit": limit,
            "total_matching": len(matching),
            "next_offset": next_offset,
            "has_more": has_more,
            "columns": columns,
            "filter": filter_name,
            "sort": sort_name,
            "direction": direction,
        }

        if not page:
            return [TextContent(type="text", text=json.dumps({
                **base_response,
                "status": "ok",
                "returned": 0,
                "rows": 0,
                "items": [],
                "message": "No matching images found.",
            }))]

        render_width = effective_thumb_width(columns, thumbnail_width)
        thumb_results = render_thumbnails(self.cli, page, render_width)
        canvas = compose_sheet(
            page, thumb_results, columns, render_width, background,
            include_filename, include_image_id, include_rating, include_sequence_number,
        )
        sheet_path = new_sheet_path()
        write_sheet(canvas, sheet_path)

        items = []
        for i, item in enumerate(page):
            image_id = str(item["id"])
            _thumb_path, error = thumb_results.get(image_id, (None, "not rendered"))
            entry = {
                "position": i + 1,
                "image_id": int(image_id),
                "filename": item.get("filename"),
                "rating": item.get("rating", 0),
                "rejected": item.get("rating", 0) == -1,
                "capture_time": item.get("capture_time") or None,
                "path": item.get("path"),
            }
            if error:
                entry["preview_status"] = "error"
                entry["preview_error"] = error
            items.append(entry)

        rows = -(-len(page) // columns)
        response = {
            **base_response,
            "status": "ok",
            "returned": len(page),
            "rows": rows,
            "sheet_path": str(sheet_path),
            "items": items,
        }
        response["sheet_url"] = self._download_url(str(sheet_path))

        out: List[Any] = []
        img = _inline_image_content(str(sheet_path), max_dim=1600, quality=88)
        if img is not None:
            out.append(img)
        out.append(TextContent(type="text", text=json.dumps(response)))
        return out

    async def _handle_rate_photos(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.bridge.call("rate_photos", arguments)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        updated = result.get("updated", 0)
        return [TextContent(
            type="text",
            text=f"Updated {updated} photos with {arguments.get('rating')} stars",
        )]

    async def _handle_tag_photo(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.bridge.call("tag_photo", arguments)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        updated = result.get("updated", 0)
        lines = [f"Tagged {updated} photos"]
        created = result.get("tags_created") or []
        if created:
            lines.append(f"created new tags: {', '.join(created)}")
        missing = result.get("missing_photos") or []
        if missing:
            lines.append(f"photo IDs not found: {', '.join(missing)}")
        return [TextContent(type="text", text="\n".join(lines))]

    async def _handle_set_photo_note(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.bridge.call("set_photo_note", arguments)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if not result.get("updated"):
            return [TextContent(
                type="text",
                text=f"Photo ID not found: {arguments.get('photo_id')}",
            )]
        return [TextContent(type="text", text=f"Note saved for photo {arguments.get('photo_id')}")]

    async def _handle_get_photo_note(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.bridge.call("get_photo_note", arguments)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if not result.get("found"):
            return [TextContent(
                type="text",
                text=f"Photo ID not found: {arguments.get('photo_id')}",
            )]
        note = result.get("note") or ""
        if not note:
            return [TextContent(type="text", text="(no note set for this photo)")]
        return [TextContent(type="text", text=note)]

    async def _handle_list_collections(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.bridge.call("list_collections", arguments)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        collections = result.get("collections") or []
        if not collections:
            return [TextContent(type="text", text="No collections (tags) found")]
        lines = [f"Found {len(collections)} collections:"]
        for c in collections:
            lines.append(f"{c['name']} ({c.get('count', 0)} photos)")
        return [TextContent(type="text", text="\n".join(lines))]

    async def _handle_list_photos_in_collection(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.bridge.call("list_photos_in_collection", arguments)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if not result.get("found"):
            return [TextContent(
                type="text",
                text=f"Collection '{arguments.get('collection')}' not found",
            )]
        photos = result.get("photos") or []
        collection_name = result.get("collection")
        if not photos:
            return [TextContent(type="text", text=f"Collection '{collection_name}' has no photos")]
        lines = [f"Collection '{collection_name}': {len(photos)} photos"]
        for p in photos:
            stars = "⭐" * (p.get("rating") or 0)
            lines.append(f"ID: {p['id']} | {p['filename']} | Rating: {stars} | {p.get('path', '')}")
        return [TextContent(type="text", text="\n".join(lines))]

    async def _handle_import_batch(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.bridge.call("import_batch", arguments)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        imported = result.get("imported", 0)
        src = result.get("source_path", arguments.get("source_path", "?"))
        return [TextContent(
            type="text",
            text=f"Imported {imported} photos from {src}",
        )]

    async def _handle_list_styles(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.bridge.call("list_styles", arguments)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        styles = result.get("styles", [])
        count = result.get("count", len(styles))
        if count == 0:
            return [TextContent(type="text", text="No styles installed.")]
        # Show count + first 50 names, with a hint if there are more.
        lines = [f"{count} styles installed:"]
        for s in styles[:50]:
            desc = s.get("description") or ""
            if desc:
                lines.append(f"  {s['name']} — {desc}")
            else:
                lines.append(f"  {s['name']}")
        if count > 50:
            lines.append(f"  ... and {count - 50} more (full list available; this is the first 50)")
        return [TextContent(type="text", text="\n".join(lines))]

    async def _handle_apply_preset(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.bridge.call("apply_preset", arguments)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        applied = result.get("applied", 0)
        missed = result.get("missed", [])
        name = result.get("preset_name", arguments.get("preset_name", "?"))
        parts = [f"Applied '{name}' to {applied} photo(s)"]
        if missed:
            parts.append(f"Missed (image not in library): {', '.join(missed)}")
        return [TextContent(type="text", text="\n".join(parts))]

    # ---- Phase 1 (T1.6): scalar darkroom editing loop ----------------------

    async def _handle_open_image_in_darkroom(self, arguments: Dict[str, Any]) -> List[TextContent]:
        image_id = arguments.get("image_id")
        if image_id is None:
            return [TextContent(type="text", text="image_id is required")]
        try:
            result = self.bridge.call("open_darkroom", {"image_id": int(image_id)}, timeout=20.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        return [TextContent(type="text", text=self._format_open_darkroom_result(image_id, result))]

    def _format_open_darkroom_result(self, image_id: Any, result: Dict[str, Any]) -> str:
        """Shared by open_image_in_darkroom and navigate_photo (both end in the
        same open_darkroom bridge call and need the same view-switch/path
        bookkeeping)."""
        view = result.get("view")
        if view != "darkroom":
            diagnostic = result.get("diagnostic")
            if diagnostic:
                # The Lua pre-check (see darktable_mcp.lua open_darkroom) caught
                # this before ever scheduling the view switch -- act_on
                # resolution (mouseover/stale active_images/collection filter)
                # would have made the switch fail silently anyway, so this is
                # a root cause, not just a timeout.
                return (
                    f"Failed to enter darkroom for image {image_id}: {diagnostic} "
                    f"(dt.gui.action_images resolved to: {result.get('action_images')})"
                )
            return (
                f"Failed to enter darkroom for image {image_id}: "
                f"still in view '{view}' after "
                f"{result.get('waited_ms_for_view_switch')}ms"
            )
        # The Lua bridge reports the path as DARKTABLE sees it, which is a
        # container path in the docker test harness (see
        # docker/run-dt-bridge.sh's bind mount) -- remap it the same way
        # dev_preview's paths are remapped so this (host) process can
        # actually hand the file to the matting sidecar in mask_raster.
        raw_path = result.get("path") or None
        self._current_image_path = _remap_bridge_path(raw_path) if raw_path else None
        bounce_note = (
            " (bounced through lighttable to force a real reload, since darkroom "
            "was already open on a different image)"
            if result.get("bounced_through_lighttable") else ""
        )
        return f"Opened image {image_id} in darkroom (view={view}){bounce_note}."

    async def _handle_navigate_photo(self, arguments: Dict[str, Any]) -> List[TextContent]:
        direction = arguments.get("direction")
        if direction not in ("next", "previous"):
            return [TextContent(type="text", text="direction must be 'next' or 'previous'")]
        sort_name = arguments.get("sort", "filename")
        sort_direction = arguments.get("direction_order", "asc")
        if sort_name not in SORT_VALUES:
            return [TextContent(type="text", text=f"sort must be one of {SORT_VALUES}")]

        try:
            current = self.bridge.call("dev_current_image", {}, timeout=15.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if not isinstance(current, dict) or not current.get("has_image"):
            return [TextContent(
                type="text",
                text="No image open in darkroom (has_image=false). Open one first via open_image_in_darkroom.",
            )]
        current_id = str(current.get("id"))

        try:
            all_images = self.bridge.call("get_collection_images", {"scope": "collection"}, timeout=10.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        ordered = sort_items(all_images, sort_name, sort_direction)
        index = next((i for i, img in enumerate(ordered) if str(img.get("id")) == current_id), None)
        if index is None:
            return [TextContent(
                type="text",
                text=(
                    f"Current image (id={current_id}) is not in the currently open "
                    "collection/filter -- can't determine its neighbor. Adjust the "
                    "lighttable filter or open an image from view_photos/get_contact_sheet."
                ),
            )]

        target_index = index - 1 if direction == "previous" else index + 1
        if target_index < 0 or target_index >= len(ordered):
            edge = "first" if direction == "previous" else "last"
            return [TextContent(
                type="text",
                text=f"Already at the {edge} photo in the collection ({index + 1}/{len(ordered)}).",
            )]

        target = ordered[target_index]
        target_id = int(target["id"])
        try:
            result = self.bridge.call("open_darkroom", {"image_id": target_id}, timeout=20.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        text = self._format_open_darkroom_result(target_id, result)
        text += f" ({direction} photo: {target_index + 1}/{len(ordered)}, {target.get('filename')})"
        return [TextContent(type="text", text=text)]

    def _resolve_current_image_path(self) -> Optional[str]:
        """Best-effort absolute path of the image open in darkroom.

        Returns the path cached by open_image_in_darkroom if we have it,
        otherwise asks darktable directly via dev_current_image (covers the
        common case of an image the user opened BY HAND in the GUI, which the
        server never saw). Caches and returns the remapped host path, or None
        if nothing is open / the bridge is unavailable.
        """
        if self._current_image_path:
            return self._current_image_path
        try:
            info = self.bridge.call("dev_current_image", {}, timeout=15.0)
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError):
            return None
        if not isinstance(info, dict) or not info.get("has_image"):
            return None
        raw_path = info.get("path") or None
        if not raw_path:
            return None
        self._current_image_path = _remap_bridge_path(raw_path)
        return self._current_image_path

    async def _handle_get_current_image(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            info = self.bridge.call("dev_current_image", {}, timeout=15.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if not isinstance(info, dict) or not info.get("has_image"):
            return [TextContent(
                type="text",
                text="No image open in darkroom (has_image=false). Open one in the GUI or call open_image_in_darkroom.",
            )]
        # Keep the path cache warm so mask_raster works without a separate call,
        # remapping the container path the same way open_image_in_darkroom does.
        raw_path = info.get("path") or None
        host_path = _remap_bridge_path(raw_path) if raw_path else None
        if host_path:
            self._current_image_path = host_path
        return [TextContent(
            type="text",
            text=(
                f"Current darkroom image: id={info.get('id')} "
                f"filename={info.get('filename')} path={host_path} "
                f"sidecar={info.get('sidecar')} "
                "(pass this exact sidecar in export_images's xmp_paths to "
                "export THIS specific duplicate/version, not the base one)"
            ),
        )]

    async def _handle_list_modules(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.bridge.call("dev_active_modules", {}, timeout=15.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        modules = result.get("modules", [])
        if not modules:
            return [TextContent(
                type="text",
                text="No active modules (is an image open in darkroom? call open_image_in_darkroom first).",
            )]
        lines = [f"{len(modules)} active module(s):"]
        for m in modules:
            flag = "enabled" if m.get("enabled") else "disabled"
            name = m.get("multi_name") or ""
            lines.append(
                f"  {m.get('op')} (instance {m.get('instance')}): {flag}"
                + (f" [{name}]" if name else "")
            )
        return [TextContent(type="text", text="\n".join(lines))]

    async def _handle_get_params(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        if not op:
            return [TextContent(type="text", text="op is required")]
        instance = int(arguments.get("instance", 0))
        try:
            result = self.bridge.call(
                "dev_get_params", {"op": op, "instance": instance}, timeout=15.0
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(type="text", text=f"get_params('{op}', {instance}): {result['error']}")]
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2),
        )]

    async def _handle_set_params(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        fields = arguments.get("fields")
        if not op:
            return [TextContent(type="text", text="op is required")]
        if not isinstance(fields, dict) or not fields:
            return [TextContent(type="text", text="fields must be a non-empty object")]
        instance = int(arguments.get("instance", 0))

        # 'enabled' is a convenience key: route it to enable_module BEFORE
        # applying the rest, and strip it out so it never reaches the native
        # dev_set_params call (where it would land in unknown_fields — the
        # C side has no 'enabled' field on any module's param struct).
        fields = dict(fields)
        enable_line: Optional[str] = None
        if "enabled" in fields:
            enabled_value = bool(fields.pop("enabled"))
            try:
                enable_result = self.bridge.call(
                    "dev_enable_module",
                    {"op": op, "instance": instance, "enabled": enabled_value},
                    timeout=15.0,
                )
            except BridgePluginNotInstalledError:
                return [TextContent(
                    type="text",
                    text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
                )]
            except BridgeTimeoutError:
                return [TextContent(
                    type="text",
                    text="darktable not running, or plugin not loaded. Open darktable and try again.",
                )]
            except BridgeError as e:
                return [TextContent(type="text", text=f"Plugin error: {e}")]
            if enable_result.get("error"):
                return [TextContent(
                    type="text",
                    text=f"set_params('{op}', {instance}): enable_module error: {enable_result['error']}",
                )]
            enable_line = f"enabled: {enable_result.get('enabled', enabled_value)}"

        if not fields:
            # 'enabled' was the only key — nothing left to write via
            # dev_set_params, report the toggle and stop here.
            lines = [f"set_params('{op}', instance={instance}): ok=True"]
            if enable_line:
                lines.append(f"  {enable_line}")
            lines.append("Call get_preview() to see the result.")
            return [TextContent(type="text", text="\n".join(lines))]

        try:
            result = self.bridge.call(
                "dev_set_params",
                {"op": op, "instance": instance, "fields": fields},
                timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(type="text", text=f"set_params('{op}', {instance}): {result['error']}")]

        lines = [f"set_params('{op}', instance={instance}): ok={result.get('ok')}"]
        if enable_line:
            lines.append(f"  {enable_line}")
        applied = result.get("applied") or {}
        if applied:
            lines.append(f"  applied: {json.dumps(applied)}")
        clamped = result.get("clamped") or []
        for c in clamped:
            lines.append(
                f"  clamped: {c.get('field')} requested {c.get('requested')} -> "
                f"applied {c.get('applied')} (bounds [{c.get('min')}, {c.get('max')}] "
                f"— already at {'max' if c.get('applied') == c.get('max') else 'min'})"
            )
        unknown = result.get("unknown_fields") or []
        if unknown:
            lines.append(f"  unknown_fields (ignored): {unknown}")
        lines.append("Call get_preview() to see the result.")
        return [TextContent(type="text", text="\n".join(lines))]

    async def _handle_get_blend_params(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        if not op:
            return [TextContent(type="text", text="op is required")]
        instance = int(arguments.get("instance", 0))
        try:
            result = self.bridge.call(
                "dev_get_blend_params", {"op": op, "instance": instance}, timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(type="text", text=f"get_blend_params('{op}', {instance}): {result['error']}")]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_set_blend_params(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        if not op:
            return [TextContent(type="text", text="op is required")]
        instance = int(arguments.get("instance", 0))
        fields: Dict[str, Any] = {}
        if arguments.get("opacity") is not None:
            fields["opacity"] = float(arguments["opacity"])
        if arguments.get("enable_uniform_blend") is not None:
            fields["enable_uniform_blend"] = bool(arguments["enable_uniform_blend"])
        if arguments.get("blend_mode") is not None:
            fields["blend_mode"] = int(arguments["blend_mode"])
        if not fields:
            return [TextContent(
                type="text",
                text="set_blend_params: at least one of opacity/enable_uniform_blend/blend_mode is required",
            )]

        try:
            result = self.bridge.call(
                "dev_set_blend_params", {"op": op, "instance": instance, "fields": fields}, timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(type="text", text=f"set_blend_params('{op}', {instance}): {result['error']}")]
        return [TextContent(
            type="text",
            text=(
                f"set_blend_params('{op}', instance={instance}): ok=True "
                f"opacity={result.get('opacity')} mask_mode={result.get('mask_mode')} "
                f"blend_mode={result.get('blend_mode')}\n"
                "Call get_preview() to see the result."
            ),
        )]

    async def _handle_list_masks(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.bridge.call("dev_list_all_masks", {}, timeout=15.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if isinstance(result, dict) and result.get("error"):
            return [TextContent(type="text", text=f"list_masks: {result['error']}")]
        masks = result if isinstance(result, list) else []
        return [TextContent(type="text", text=json.dumps({"masks": masks}, indent=2))]

    async def _handle_get_module_mask(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        if not op:
            return [TextContent(type="text", text="op is required")]
        instance = int(arguments.get("instance", 0))
        try:
            blend = self.bridge.call(
                "dev_get_blend_params", {"op": op, "instance": instance}, timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if blend.get("error"):
            return [TextContent(type="text", text=f"get_module_mask('{op}', {instance}): {blend['error']}")]

        try:
            shapes_result = self.bridge.call(
                "dev_list_masks", {"op": op, "instance": instance}, timeout=15.0,
            )
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            return [TextContent(type="text", text=f"get_module_mask('{op}', {instance}): {e}")]

        if isinstance(shapes_result, dict) and shapes_result.get("error"):
            return [TextContent(
                type="text", text=f"get_module_mask('{op}', {instance}): {shapes_result['error']}",
            )]
        shapes = shapes_result if isinstance(shapes_result, list) else []

        response = {
            "op": op,
            "instance": instance,
            "opacity": blend.get("opacity"),
            "mask_mode": blend.get("mask_mode"),
            "blend_mode": blend.get("blend_mode"),
            "invert": blend.get("invert"),
            "shapes": shapes,
        }
        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    async def _handle_get_mask_geometry(self, arguments: Dict[str, Any]) -> List[TextContent]:
        mask_id = arguments.get("mask_id")
        if mask_id is None:
            return [TextContent(type="text", text="mask_id is required")]
        try:
            result = self.bridge.call(
                "dev_get_mask", {"mask_id": int(mask_id)}, timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(type="text", text=f"get_mask_geometry({mask_id}): {result['error']}")]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_rename_mask(self, arguments: Dict[str, Any]) -> List[TextContent]:
        mask_id = arguments.get("mask_id")
        name = arguments.get("name")
        if mask_id is None:
            return [TextContent(type="text", text="mask_id is required")]
        if not name:
            return [TextContent(type="text", text="name is required")]
        try:
            result = self.bridge.call(
                "dev_rename_mask", {"mask_id": int(mask_id), "name": str(name)}, timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(type="text", text=f"rename_mask({mask_id}): {result['error']}")]
        return [TextContent(
            type="text",
            text=f"rename_mask: ok=True mask_id={result.get('mask_id')} name={result.get('name')!r}",
        )]

    async def _handle_delete_mask(self, arguments: Dict[str, Any]) -> List[TextContent]:
        mask_id = arguments.get("mask_id")
        if mask_id is None:
            return [TextContent(type="text", text="mask_id is required")]
        try:
            result = self.bridge.call(
                "dev_delete_mask", {"mask_id": int(mask_id)}, timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(type="text", text=f"delete_mask({mask_id}): {result['error']}")]
        return [TextContent(
            type="text", text=f"delete_mask: ok=True mask_id={result.get('mask_id')}",
        )]

    async def _handle_attach_mask(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        if not op:
            return [TextContent(type="text", text="op is required")]
        instance = int(arguments.get("instance", 0))
        formid = arguments.get("formid")
        if formid is None:
            return [TextContent(type="text", text="attach_mask: formid is required")]
        operation = arguments.get("operation") or "union"
        try:
            result = self.bridge.call(
                "dev_attach_mask",
                {"op": op, "instance": instance, "formid": int(formid), "operation": operation},
                timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(
                type="text", text=f"attach_mask('{op}', {instance}, formid={formid}): {result['error']}",
            )]

        lines = [
            f"attach_mask('{op}', instance={instance}): formid={result.get('formid')} "
            f"operation={result.get('operation')}"
        ]
        lines.append(self._verify_drawn_mask_enabled(op, instance, result.get("mask_mode")))
        lines.append("Call get_preview() to see the result.")
        return [TextContent(type="text", text="\n".join(lines))]

    def _verify_drawn_mask_enabled(self, op: str, instance: int, reported_mask_mode: Any) -> str:
        """Read blend_params back from the bridge (a SEPARATE call, not just
        trusting the mask_mode attach_mask/set_module_mask already reported)
        and confirm DEVELOP_MASK_MASK actually stuck -- attaching a shape to
        the group used to leave the mask inert until the user clicked the
        pencil icon by hand (bugreport 2026-07-27). Returns a one-line status
        string; never raises."""
        try:
            blend = self.bridge.call(
                "dev_get_blend_params", {"op": op, "instance": instance}, timeout=15.0,
            )
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            return f"  WARNING: could not verify mask_mode after attach: {e}"
        if blend.get("error"):
            return f"  WARNING: could not verify mask_mode after attach: {blend['error']}"
        mask_mode = blend.get("mask_mode")
        if not isinstance(mask_mode, int) or not (mask_mode & _DEVELOP_MASK_MASK):
            return (
                f"  WARNING: drawn mask is NOT enabled after attach (mask_mode={mask_mode}, "
                f"reported={reported_mask_mode}) -- the mask exists but will have NO visible "
                "effect until enabled; this should not happen, report it"
            )
        return f"  verified: drawn mask enabled (mask_mode={mask_mode})"

    async def _handle_detach_mask(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        if not op:
            return [TextContent(type="text", text="op is required")]
        instance = int(arguments.get("instance", 0))
        formid = arguments.get("formid")
        if formid is None:
            return [TextContent(type="text", text="detach_mask: formid is required")]
        try:
            result = self.bridge.call(
                "dev_detach_mask", {"op": op, "instance": instance, "formid": int(formid)}, timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(
                type="text", text=f"detach_mask('{op}', {instance}, formid={formid}): {result['error']}",
            )]
        return [TextContent(
            type="text",
            text=(
                f"detach_mask('{op}', instance={instance}): formid={result.get('formid')} detached\n"
                "Call get_preview() to see the result."
            ),
        )]

    async def _handle_set_module_mask(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        if not op:
            return [TextContent(type="text", text="op is required")]
        instance = int(arguments.get("instance", 0))
        shapes = arguments.get("shapes")
        if not isinstance(shapes, list):
            return [TextContent(type="text", text="set_module_mask: shapes must be a list")]

        desired: Dict[int, str] = {}
        for s in shapes:
            if not isinstance(s, dict) or s.get("formid") is None:
                return [TextContent(type="text", text="set_module_mask: each shape needs a formid")]
            desired[int(s["formid"])] = s.get("operation") or "union"

        def call(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
            try:
                result = self.bridge.call(method, params, timeout=15.0)
            except BridgePluginNotInstalledError:
                return {"error": "darktable-mcp plugin not installed. Run: darktable-mcp install-plugin"}
            except BridgeTimeoutError:
                return {"error": "darktable not running, or plugin not loaded. Open darktable and try again."}
            except BridgeError as e:
                return {"error": f"Plugin error: {e}"}
            return result if isinstance(result, dict) else {"_list": result}

        current_result = call("dev_list_masks", {"op": op, "instance": instance})
        if current_result.get("error"):
            return [TextContent(
                type="text", text=f"set_module_mask('{op}', {instance}): {current_result['error']}",
            )]
        current_list = current_result.get("_list") or []
        current: Dict[int, str] = {
            int(s["mask_id"]): s.get("operation") or "union" for s in current_list
        }

        detached: List[int] = []
        attached: List[int] = []
        errors: List[str] = []

        # Shapes to drop, and shapes whose operation changed: attach_mask
        # cannot rewrite an already-attached shape's operation in place (the
        # underlying dt_masks_group_add_form would append a SECOND, duplicate
        # point for the same formid rather than updating the existing one) --
        # so a changed operation goes through detach+reattach, same as a
        # plain removal.
        for formid, current_op in current.items():
            if formid not in desired or desired[formid] != current_op:
                result = call("dev_detach_mask", {"op": op, "instance": instance, "formid": formid})
                if result.get("error"):
                    errors.append(f"detach formid={formid}: {result['error']}")
                else:
                    detached.append(formid)

        for formid, wanted_op in desired.items():
            if formid not in current or current[formid] != wanted_op:
                result = call("dev_attach_mask", {
                    "op": op, "instance": instance, "formid": formid, "operation": wanted_op,
                })
                if result.get("error"):
                    errors.append(f"attach formid={formid}: {result['error']}")
                else:
                    attached.append(formid)

        fields: Dict[str, Any] = {}
        if arguments.get("opacity") is not None:
            fields["opacity"] = float(arguments["opacity"])
        if arguments.get("invert") is not None:
            fields["invert"] = bool(arguments["invert"])
        blend_result = None
        if fields:
            blend_result = call("dev_set_blend_params", {"op": op, "instance": instance, "fields": fields})
            if blend_result.get("error"):
                errors.append(f"set_blend_params: {blend_result['error']}")

        lines = [f"set_module_mask('{op}', instance={instance}): attached={attached} detached={detached}"]
        if blend_result and not blend_result.get("error"):
            lines.append(f"  opacity={blend_result.get('opacity')} invert={blend_result.get('invert')}")
        if attached:
            lines.append(self._verify_drawn_mask_enabled(op, instance, None))
        if errors:
            lines.append("  errors: " + "; ".join(errors))
        lines.append("Call get_module_mask() to confirm the final state.")
        return [TextContent(type="text", text="\n".join(lines))]

    def _register_download(self, host_path: str) -> Optional[str]:
        """Register an absolute file path for HTTP download and return its token.

        Returns None if no public URL is configured (nothing to build a link
        from) or the file is missing. Token is unguessable; only paths we
        register are ever served, so /mcp/files/<token> can't be used to read
        arbitrary files. FIFO-evicts once the cap is hit.
        """
        if not os.environ.get("DTMCP_PUBLIC_URL"):
            return None
        real = os.path.realpath(host_path)
        if not os.path.isfile(real):
            return None
        token = uuid.uuid4().hex
        self._download_registry[token] = real
        while len(self._download_registry) > self._download_registry_cap:
            self._download_registry.popitem(last=False)
        return token

    def _download_url(self, host_path: str) -> Optional[str]:
        """Public https URL a remote client can stream the file from, or None
        when no public URL is set. Pairs with the /mcp/files route."""
        base = os.environ.get("DTMCP_PUBLIC_URL")
        token = self._register_download(host_path)
        if not base or not token:
            return None
        return f"{base.rstrip('/')}/mcp/files/{token}"

    def _store_viewport_snapshot(self, record: Dict[str, Any]) -> str:
        """Store a capture_viewport result and return its snapshot_id.
        FIFO-evicts on cap, same pattern as _register_download."""
        snapshot_id = "vp_" + uuid.uuid4().hex
        record["created_at"] = time.monotonic()
        self._viewport_snapshots[snapshot_id] = record
        while len(self._viewport_snapshots) > self._viewport_snapshot_cap:
            self._viewport_snapshots.popitem(last=False)
        return snapshot_id

    def _get_viewport_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """Look up a snapshot by id, raising ViewportCoordinateError if it's
        missing or past its TTL -- never silently falls back to "wherever
        the viewport happens to be now"."""
        record = self._viewport_snapshots.get(snapshot_id) if snapshot_id else None
        if record is None:
            raise ViewportCoordinateError(
                f"snapshot_id {snapshot_id!r} not found (never captured, or "
                "server restarted since) -- call capture_viewport again"
            )
        age = time.monotonic() - record["created_at"]
        if age > self._viewport_snapshot_ttl_s:
            del self._viewport_snapshots[snapshot_id]
            raise ViewportCoordinateError(
                f"snapshot_id {snapshot_id!r} expired ({age:.0f}s old, "
                f"limit {self._viewport_snapshot_ttl_s:.0f}s) -- the viewport "
                "may have moved since; call capture_viewport again"
            )
        return record

    def _darkroom_image(self) -> Dict[str, Any]:
        """Identity of the image currently open in darkroom:
        {"has_image": True, "id": int, "filename": str, "path": str}, or
        {"has_image": False, "error": "..."} when nothing is loaded / the
        bridge call itself failed.

        EVERY darkroom binding resolves against darktable's GLOBAL state
        (darktable.develop) -- there is no per-call image handle anywhere in
        the C API -- so a snapshot and the edit made from it stay on the same
        photo only if we read this identity and compare it ourselves. Without
        that check a view switch between two consecutive tool calls silently
        retouches whatever image happens to be open (2026-07-26 bugreport)."""
        try:
            result = self.bridge.call("dev_current_image", {}, timeout=10.0)
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            return {"has_image": False, "error": str(e)}
        if not isinstance(result, dict):
            return {"has_image": False, "error": f"unexpected reply: {result!r}"}
        if result.get("error"):
            return {"has_image": False, "error": str(result["error"])}
        if not result.get("has_image"):
            result.setdefault("error", "no darkroom image loaded")
        return result

    @staticmethod
    def _image_label(img: Dict[str, Any]) -> str:
        if not img or not img.get("has_image"):
            return f"none ({(img or {}).get('error', 'no darkroom image loaded')})"
        return f"id={img.get('id')} filename={img.get('filename')}"

    def _snapshot_image_guard(self, snapshot: Dict[str, Any], tool: str):
        """Refuse to mutate when the darkroom no longer holds the image the
        snapshot was captured from. Returns an error response list, or None
        when it is safe to proceed."""
        expected = snapshot.get("image") or {}
        current = self._darkroom_image()
        if not current.get("has_image"):
            return [TextContent(type="text", text=(
                f"{tool}: darkroom image mismatch -- snapshot was captured on "
                f"{self._image_label(expected)}, but darktable has no image open "
                f"now ({current.get('error')}). Nothing was written. Reopen that "
                "image in darkroom and call capture_viewport again."
            ))]
        if expected.get("id") is not None and current.get("id") != expected.get("id"):
            return [TextContent(type="text", text=(
                f"{tool}: darkroom image mismatch -- snapshot image_id="
                f"{expected.get('id')} ({expected.get('filename')}), active "
                f"image_id={current.get('id')} ({current.get('filename')}). "
                "Nothing was written -- the coordinates in the snapshot mean "
                "nothing on a different photo. Reopen the snapshot's image, or "
                "call capture_viewport again for the active one."
            ))]
        return None

    def _explain_darkroom_error(self, message: str) -> str:
        """Append the live darkroom state to an error that reports the
        darkroom as gone. The C bindings say "no darkroom image loaded"
        whenever dev->iop is NULL at that instant -- i.e. darktable had left
        darkroom, or was mid-switch to another image. Naming the image that IS
        open turns an unexplained failure into a diagnosis (2026-07-26
        bugreport: a delete failed this way one call after a successful
        add)."""
        text = str(message)
        if "no darkroom image loaded" not in text and "not in darkroom" not in text:
            return text
        return (
            f"{text} -- darkroom state right now: "
            f"{self._image_label(self._darkroom_image())}. darktable left "
            "darkroom or switched image between calls; the shape itself may "
            "still exist on the image it was added to."
        )

    def _render_snapshot_preview(self, snapshot: Dict[str, Any]):
        """Re-render the exact region+size of a captured snapshot, for a
        before/after comparison after an in-viewport retouch edit. Returns
        (ImageContent|None, text_line) -- failures degrade to a text-only
        line rather than failing the whole tool call, since the edit itself
        already succeeded by the time this runs."""
        render = snapshot["render"]
        params = {
            "max_w": render.get("width") or 1400,
            "max_h": render.get("height") or 1400,
            "region": snapshot["region"],
        }
        try:
            result = self.bridge.call("dev_preview", params, timeout=20.0)
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            return None, f"  post-edit preview unavailable: {e}"
        if result.get("error"):
            return None, f"  post-edit preview unavailable: {result['error']}"
        path = result.get("path") or result.get("stale_preview")
        if not path:
            return None, "  post-edit preview unavailable: no path in result"
        host_path = _remap_bridge_path(path)

        # dev_preview renders whatever image darkroom holds RIGHT NOW; it
        # carries no image identity of its own. Re-read the identity and say
        # which photo the returned frame is of, so a preview from a different
        # image can never be mistaken for "the edit I just made" (the
        # 2026-07-26 bugreport: a snapshot of a b&w portrait came back with a
        # post-edit preview of an unrelated colour photo).
        expected = snapshot.get("image") or {}
        current = self._darkroom_image()
        line = f"  post-edit preview: {host_path}"
        if current.get("has_image"):
            line += f" (preview_image_id={current.get('id')} {current.get('filename')})"
        if expected.get("id") is not None and current.get("id") != expected.get("id"):
            line = (
                f"  WARNING: post-edit preview is NOT the snapshot's image -- "
                f"snapshot image_id={expected.get('id')} "
                f"({expected.get('filename')}), preview image "
                f"{self._image_label(current)}. The darkroom image changed "
                f"during this call; ignore this preview.\n" + line
            )
        return _inline_image_content(host_path), line

    def _backtransform_to_mask_space(
        self,
        display_target: Dict[str, float],
        display_source: Dict[str, float],
        display_radius: float,
        display_feather: float,
    ):
        """Second stage of the viewport-relative retouch pipeline: convert
        PROCESSED/DISPLAY-frame-normalized coordinates (viewport_coords.py's
        output -- the frame get_viewport()/get_preview() use) into the
        PIPE-INPUT/mask-frame-normalized coordinates retouch_add_shape/
        retouch_update_shape actually store (see dt.develop.backtransform_point's
        doc comment, src-dt/src/lua/develop.c, for why these frames differ).

        Returns (mask_dict, error_response) -- exactly one is None. mask_dict
        has target/source ({x,y}) and radius/feather (floats)."""
        target_params: Dict[str, Any] = {
            "x": display_target["x"], "y": display_target["y"], "len1": display_radius,
        }
        if display_feather > 0:
            target_params["len2"] = display_feather
        try:
            target_result = self.bridge.call("dev_backtransform_point", target_params, timeout=15.0)
        except BridgePluginNotInstalledError:
            return None, [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return None, [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return None, [TextContent(type="text", text=f"Plugin error: {e}")]
        if target_result.get("error"):
            return None, [TextContent(
                type="text", text=f"backtransform_point(target): {target_result['error']}"
            )]

        try:
            source_result = self.bridge.call(
                "dev_backtransform_point",
                {"x": display_source["x"], "y": display_source["y"]},
                timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return None, [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return None, [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return None, [TextContent(type="text", text=f"Plugin error: {e}")]
        if source_result.get("error"):
            return None, [TextContent(
                type="text", text=f"backtransform_point(source): {source_result['error']}"
            )]

        return {
            "target": {"x": target_result["x"], "y": target_result["y"]},
            "source": {"x": source_result["x"], "y": source_result["y"]},
            "radius": target_result.get("len1", display_radius),
            "feather": target_result.get("len2", 0.0) if display_feather > 0 else 0.0,
        }, None

    def _backtransform_polygon_to_mask_space(
        self, polygon_display: List[Dict[str, float]]
    ):
        """mask_object's segmentation sidecar returns a polygon normalized
        against dev_preview's PROCESSED/DISPLAY frame (it segments that
        render directly). add_path_mask stores points in the PIPE-INPUT/
        mask frame instead -- the exact same mismatch already fixed for
        retouch's target/source points (_backtransform_to_mask_space above;
        see dt.develop.backtransform_point's doc comment in
        src-dt/src/lua/develop.c for why the two frames differ, e.g. any
        portrait EXIF orientation swaps width/height between them).

        Bugreport 2026-07-27 (SAM2 body mask landing on a tree instead of
        the subject) traced to this exact gap: mask_object wrote the
        sidecar's display-frame polygon straight into add_path_mask with a
        comment claiming no rescaling was needed. It was needed.

        Backtransforms every vertex individually via the same
        dev_backtransform_point bridge call retouch already uses (no C
        change required -- the binding has no batch form yet, so this is
        one bridge round-trip per vertex, ~10-30 total).

        Returns (mask_frame_points, None) or (None, error_response)."""
        mask_points: List[Dict[str, float]] = []
        for i, pt in enumerate(polygon_display):
            try:
                result = self.bridge.call(
                    "dev_backtransform_point", {"x": pt["x"], "y": pt["y"]}, timeout=15.0
                )
            except BridgePluginNotInstalledError:
                return None, [TextContent(
                    type="text",
                    text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
                )]
            except BridgeTimeoutError:
                return None, [TextContent(
                    type="text",
                    text="darktable not running, or plugin not loaded. Open darktable and try again.",
                )]
            except BridgeError as e:
                return None, [TextContent(type="text", text=f"Plugin error: {e}")]
            if result.get("error"):
                return None, [TextContent(
                    type="text",
                    text=f"mask_object: backtransform_point(polygon[{i}]): {result['error']}",
                )]
            mask_points.append({"x": result["x"], "y": result["y"]})
        return mask_points, None

    async def _handle_get_preview(self, arguments: Dict[str, Any]) -> List[Any]:
        max_w = int(arguments.get("max_w", 1024))
        max_h = int(arguments.get("max_h", 1024))
        # Deterministic inline delivery (bug: preview sometimes not shown).
        return_image = bool(arguments.get("return_image", True))
        inline_max_dim = int(arguments.get("inline_max_dim", 1024))
        params: Dict[str, Any] = {"max_w": max_w, "max_h": max_h}
        region = arguments.get("region")
        if isinstance(region, dict) and all(k in region for k in ("x", "y", "w", "h")):
            params["region"] = {
                "x": float(region["x"]),
                "y": float(region["y"]),
                "w": float(region["w"]),
                "h": float(region["h"]),
            }
        try:
            result = self.bridge.call("dev_preview", params, timeout=20.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(type="text", text=f"get_preview: {result['error']}")]

        status = result.get("status")
        path = result.get("path") or result.get("stale_preview")
        if not path:
            return [TextContent(type="text", text=f"get_preview: no path in result: {result}")]
        host_path = _remap_bridge_path(path)
        lines = [f"status: {status}", f"path: {host_path}"]
        # Which photo this frame is of, plus WHY its size is what it is: the
        # render is a crop of the processed frame (frame_width/height), capped
        # at max_w/max_h and never upscaled past the crop's own resolution --
        # so a small region legitimately returns a small image. Reported
        # because bare, varying dimensions read as a bug (2026-07-26 report).
        img = self._darkroom_image()
        lines.append(f"image: {self._image_label(img)}")
        if result.get("width") and result.get("height"):
            lines.append(f"size: {result['width']}x{result['height']}")
        if result.get("frame_width") and result.get("frame_height"):
            lines.append(
                f"processed frame: {result['frame_width']}x{result['frame_height']}"
            )
        rendered_region = result.get("region")
        if isinstance(rendered_region, dict):
            lines.append(f"rendered region: {json.dumps(rendered_region)}")
        else:
            lines.append("rendered region: full frame (no region requested)")
        lines.append(
            f"size rule: crop of the processed frame, capped at "
            f"max_w={max_w}/max_h={max_h}, never upscaled past the crop"
        )
        if status == "processing":
            lines.append("(pipe was still processing; this is the last good frame)")
        # Full-res download URL for a remote client that would rather stream the
        # PNG than parse a big base64 blob. Served by the HTTP transport's
        # /mcp/files route (same nginx /mcp vhost; no Bearer needed, the
        # per-file token in the URL is its own capability, see _register_download).
        dl = self._download_url(host_path)
        if dl:
            lines.append(f"url: {dl} (no auth header needed, token in URL is the credential)")

        # Inline a SMALL re-encoded image so a remote client can SEE the render,
        # not just a local path it can't read. Downscaled JPEG (not the raw PNG)
        # so it always fits the client's inline-image budget -- a full-size PNG
        # is ~1MB base64 and gets silently dropped. Image first (what the model
        # looks at), metadata text second. return_image=false -> metadata only.
        out: List[Any] = []
        if return_image:
            img = _inline_image_content(host_path, max_dim=inline_max_dim)
            if img is not None:
                out.append(img)
                lines.append(
                    f"(inline image: JPEG, downscaled to <= {inline_max_dim}px; "
                    "full-res PNG at path/url above)"
                )
            else:
                lines.append(
                    "(preview file not readable from the server host -- image not inlined)"
                )
        out.append(TextContent(type="text", text="\n".join(lines)))
        return out

    async def _handle_enable_module(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        if not op:
            return [TextContent(type="text", text="op is required")]
        enabled = arguments.get("enabled")
        if enabled is None:
            return [TextContent(type="text", text="enabled is required")]
        instance = int(arguments.get("instance", 0))
        try:
            result = self.bridge.call(
                "dev_enable_module",
                {"op": op, "instance": instance, "enabled": bool(enabled)},
                timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(type="text", text=f"enable_module('{op}', {instance}): {result['error']}")]
        return [TextContent(
            type="text",
            text=f"enable_module('{op}', instance={instance}): enabled={result.get('enabled', bool(enabled))}",
        )]

    async def _handle_list_luts(self, arguments: Dict[str, Any]) -> List[TextContent]:
        directory = arguments.get("directory") or None
        try:
            root_dir = self.bridge.call(
                "dev_get_conf_string",
                {"key": "plugins/darkroom/lut3d/def_path"},
                timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        try:
            items = scan_lut_directory(root_dir, directory)
        except (LutRootNotConfiguredError, FileNotFoundError, ValueError) as e:
            return [TextContent(type="text", text=f"list_luts: {e}")]

        root = Path(root_dir).expanduser().resolve()
        for item in items:
            if item["format"] == "cube":
                header = parse_cube_header(root / item["path"])
                if "title" in header:
                    item["title"] = header["title"]
                if "size" in header:
                    item["size"] = header["size"]

        if not items:
            scope = f" under {directory!r}" if directory else ""
            return [TextContent(
                type="text",
                text=f"list_luts: no .cube/.3dl/.png files found{scope} in {root}",
            )]

        lines = [f"list_luts: {len(items)} file(s) under {root}"]
        if directory:
            lines[0] += f" (scoped to {directory!r})"
        lines.append(json.dumps(items, indent=2))
        return [TextContent(type="text", text="\n".join(lines))]

    async def _lut3d_preview_once(
        self,
        instance: int,
        path: str,
        colorspace: Optional[str],
        interpolation: Optional[str],
        max_w: int,
        max_h: int,
        region: Optional[Dict[str, Any]],
        opacity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Apply `path` to lut3d instance `instance`, render a preview, then
        restore lut3d's prior filepath/colorspace/interpolation/lutname/
        enabled/blend-opacity state -- the shared core of preview_lut and
        compare_luts (each LUT in a compare_luts call goes through this
        exactly once, one at a time -- never concurrently, since they all
        mutate the same module).

        `opacity`, if given, is 0..1 (mask_object/retouch's MCP-facing
        convention) and is converted to blend_params' native 0..100 scale
        here -- lut3d has no "amount" field of its own, so partial-strength
        LUTs (the reported 15-50% portrait use case) only work through blend
        opacity. Applying it always sets enable_uniform_blend=true (plain
        blend, no mask) and is restored to the module's PRIOR blend state
        (opacity + mask_mode) exactly like the params restore below.

        Bridge-transport failures (BridgePluginNotInstalledError,
        BridgeTimeoutError, and BridgeError generally) PROPAGATE to the
        caller, which should catch them ONCE around the whole operation: a
        transport failure means darktable itself is unreachable, not that
        this one LUT is bad, so retrying it per-item would be pointless.
        Anything else (instance not found, bad params, render error) comes
        back as {"ok": False, "error": ...} instead of raising, so a caller
        looping over several LUTs (compare_luts) can keep going after one
        fails.
        """
        active = self.bridge.call("dev_active_modules", {}, timeout=15.0)
        modules = active.get("modules", [])
        match = next(
            (m for m in modules if m.get("op") == "lut3d" and int(m.get("instance", 0)) == instance),
            None,
        )
        if match is None:
            return {"ok": False, "error": (
                f"lut3d instance {instance} not found -- call add_instance('lut3d') "
                "first if instance > 0, or open an image in darkroom first"
            )}
        was_enabled = bool(match.get("enabled"))

        snap = self.bridge.call("dev_get_params", {"op": "lut3d", "instance": instance}, timeout=15.0)
        if snap.get("error"):
            return {"ok": False, "error": f"get_params: {snap['error']}"}
        fields = snap.get("fields", {})
        original_fields: Dict[str, Any] = {}
        if isinstance(fields.get("filepath"), str):
            original_fields["filepath"] = fields["filepath"]
        if isinstance(fields.get("lutname"), str):
            original_fields["lutname"] = fields["lutname"]
        for key in ("colorspace", "interpolation"):
            val = fields.get(key)
            if isinstance(val, dict) and "label" in val:
                original_fields[key] = val["label"]

        original_blend: Optional[Dict[str, Any]] = None
        if opacity is not None:
            blend_snap = self.bridge.call(
                "dev_get_blend_params", {"op": "lut3d", "instance": instance}, timeout=15.0,
            )
            if blend_snap.get("error"):
                return {"ok": False, "error": f"get_blend_params: {blend_snap['error']}"}
            snap_mask_mode = blend_snap.get("mask_mode")
            if isinstance(snap_mask_mode, int) and snap_mask_mode not in (0, 1):
                # enable_uniform_blend (the only write path available here) is
                # ALL-OR-NOTHING on mask_mode -- restoring it later would
                # clobber a drawn/parametric/raster mask already wired to this
                # instance. Refuse rather than silently destroy that setup;
                # use set_blend_params directly if this is deliberate.
                return {"ok": False, "error": (
                    f"lut3d instance {instance} already has a mask configured "
                    f"(mask_mode={snap_mask_mode}) -- opacity via preview_lut/"
                    "compare_luts only supports the plain no-mask case. Use "
                    "set_blend_params directly if this is intentional."
                )}
            original_blend = {
                "opacity": blend_snap.get("opacity"),
                "mask_mode": snap_mask_mode,
            }

        new_fields: Dict[str, Any] = {"filepath": path, "enabled": True}
        if colorspace:
            new_fields["colorspace"] = colorspace
        if interpolation:
            new_fields["interpolation"] = interpolation

        set_result = self.bridge.call(
            "dev_set_params", {"op": "lut3d", "instance": instance, "fields": new_fields}, timeout=15.0,
        )
        if set_result.get("error"):
            return {"ok": False, "error": f"set_params: {set_result['error']}"}
        clamped = set_result.get("clamped") or []

        if opacity is not None:
            blend_set = self.bridge.call(
                "dev_set_blend_params",
                {
                    "op": "lut3d", "instance": instance,
                    "fields": {
                        "opacity": max(0.0, min(1.0, opacity)) * 100.0,
                        "enable_uniform_blend": True,
                    },
                },
                timeout=15.0,
            )
            if blend_set.get("error"):
                # lut3d params were already applied above -- restore them
                # before reporting, same as any other failure past this point.
                restore_fields = dict(original_fields)
                restore_fields["enabled"] = was_enabled
                self.bridge.call(
                    "dev_set_params", {"op": "lut3d", "instance": instance, "fields": restore_fields}, timeout=15.0,
                )
                return {"ok": False, "error": f"set_blend_params: {blend_set['error']}"}

        preview_error: Optional[str] = None
        host_path: Optional[str] = None
        try:
            preview_params: Dict[str, Any] = {"max_w": max_w, "max_h": max_h}
            if region:
                preview_params["region"] = region
            preview_result = self.bridge.call("dev_preview", preview_params, timeout=20.0)
            if preview_result.get("error"):
                preview_error = preview_result["error"]
            else:
                raw_path = preview_result.get("path") or preview_result.get("stale_preview")
                if raw_path:
                    host_path = _remap_bridge_path(raw_path)
        finally:
            restore_fields = dict(original_fields)
            restore_fields["enabled"] = was_enabled
            restore_result = self.bridge.call(
                "dev_set_params", {"op": "lut3d", "instance": instance, "fields": restore_fields}, timeout=15.0,
            )
            restore_warning = (
                f"restore failed, lut3d may be left in the applied state: {restore_result['error']}"
                if restore_result.get("error") else None
            )
            if original_blend is not None:
                blend_restore = self.bridge.call(
                    "dev_set_blend_params",
                    {
                        "op": "lut3d", "instance": instance,
                        "fields": {
                            "opacity": original_blend["opacity"],
                            # Snapshot-time guard above already refused
                            # anything but mask_mode 0/1, so this is exactly
                            # the module's prior on/off state, never a mask.
                            "enable_uniform_blend": original_blend["mask_mode"] == 1,
                        },
                    },
                    timeout=15.0,
                )
                if blend_restore.get("error") and not restore_warning:
                    restore_warning = (
                        f"blend restore failed, lut3d opacity may be left applied: {blend_restore['error']}"
                    )

        if preview_error:
            return {"ok": False, "error": f"get_preview: {preview_error}", "restore_warning": restore_warning}
        return {"ok": True, "host_path": host_path, "clamped": clamped, "restore_warning": restore_warning}

    async def _handle_preview_lut(self, arguments: Dict[str, Any]) -> List[Any]:
        path = arguments.get("path")
        if not path:
            return [TextContent(type="text", text="path is required")]
        instance = int(arguments.get("instance", 0))
        colorspace = arguments.get("colorspace")
        interpolation = arguments.get("interpolation")
        max_w = int(arguments.get("max_w", 1024))
        max_h = int(arguments.get("max_h", 1024))
        region = arguments.get("region")
        if not (isinstance(region, dict) and all(k in region for k in ("x", "y", "w", "h"))):
            region = None
        opacity = arguments.get("opacity")
        opacity = float(opacity) if opacity is not None else None

        try:
            root_dir = self.bridge.call(
                "dev_get_conf_string", {"key": "plugins/darkroom/lut3d/def_path"}, timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        try:
            resolve_lut_path(root_dir, path)
        except (LutRootNotConfiguredError, FileNotFoundError, ValueError) as e:
            return [TextContent(type="text", text=f"preview_lut: {e}")]

        try:
            result = await self._lut3d_preview_once(
                instance, path, colorspace, interpolation, max_w, max_h, region, opacity,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if not result["ok"]:
            lines = [f"preview_lut('{path}'): {result['error']}"]
            if result.get("restore_warning"):
                lines.append(f"WARNING: {result['restore_warning']}")
            return [TextContent(type="text", text="\n".join(lines))]

        lines = [f"preview_lut('{path}'): ok=True (lut3d restored to its prior state)"]
        if result["clamped"]:
            lines.append(f"clamped/rejected fields: {json.dumps(result['clamped'])}")
        if result.get("restore_warning"):
            lines.append(f"WARNING: {result['restore_warning']}")

        out: List[Any] = []
        if result["host_path"]:
            lines.append(f"path: {result['host_path']}")
            img = _inline_image_content(result["host_path"])
            if img is not None:
                out.append(img)
        out.append(TextContent(type="text", text="\n".join(lines)))
        return out

    async def _handle_compare_luts(self, arguments: Dict[str, Any]) -> List[Any]:
        paths = arguments.get("paths")
        if not isinstance(paths, list) or not paths:
            return [TextContent(type="text", text="paths must be a non-empty array")]
        instance = int(arguments.get("instance", 0))
        colorspace = arguments.get("colorspace")
        interpolation = arguments.get("interpolation")
        opacity = arguments.get("opacity")
        opacity = float(opacity) if opacity is not None else None
        columns = max(1, int(arguments.get("columns", 4)))
        requested_width = int(arguments.get("cell_width", 480))
        background = arguments.get("background", "dark")
        if background not in ("dark", "light"):
            background = "dark"
        cell_width = effective_cell_width(columns, requested_width)
        # Aspect-preserving cell height cap (typical 3:2 photo) -- never
        # upscaled beyond the render's own resolution, same as get_preview.
        cell_height = int(cell_width * 0.75)

        try:
            root_dir = self.bridge.call(
                "dev_get_conf_string", {"key": "plugins/darkroom/lut3d/def_path"}, timeout=15.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        entries: List[Dict[str, Any]] = []
        try:
            for path in paths:
                try:
                    resolve_lut_path(root_dir, path)
                except (LutRootNotConfiguredError, FileNotFoundError, ValueError) as e:
                    entries.append({"label": path, "image_path": None, "error": str(e)})
                    continue
                result = await self._lut3d_preview_once(
                    instance, path, colorspace, interpolation, cell_width, cell_height, None, opacity,
                )
                if result["ok"]:
                    entries.append({
                        "label": path,
                        "image_path": result["host_path"],
                        "error": None,
                        "restore_warning": result.get("restore_warning"),
                    })
                else:
                    entries.append({
                        "label": path,
                        "image_path": None,
                        "error": result["error"],
                        "restore_warning": result.get("restore_warning"),
                    })
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        canvas = compose_lut_compare_grid(entries, columns, cell_width, background)
        sheet_path = new_sheet_path()
        write_sheet(canvas, sheet_path)

        succeeded = sum(1 for e in entries if not e["error"])
        response = {
            "status": "ok",
            "requested": len(paths),
            "succeeded": succeeded,
            "failed": len(paths) - succeeded,
            "columns": columns,
            "sheet_path": str(sheet_path),
            "sheet_url": self._download_url(str(sheet_path)),
            "items": [
                {
                    "path": e["label"],
                    "ok": not e["error"],
                    "error": e["error"],
                    "restore_warning": e.get("restore_warning"),
                }
                for e in entries
            ],
        }

        out: List[Any] = []
        img = _inline_image_content(str(sheet_path), max_dim=1600, quality=88)
        if img is not None:
            out.append(img)
        out.append(TextContent(type="text", text=json.dumps(response, indent=2)))
        return out

    async def _handle_add_instance(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        if not op:
            return [TextContent(type="text", text="op is required")]
        fields = arguments.get("fields")
        if fields is not None and not isinstance(fields, dict):
            return [TextContent(type="text", text="add_instance: fields must be an object")]
        payload: Dict[str, Any] = {"op": op}
        if fields:
            payload["fields"] = fields
        try:
            result = self.bridge.call("dev_add_instance", payload, timeout=15.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(type="text", text=f"add_instance('{op}'): {result['error']}")]
        lines = [
            f"add_instance('{op}'): new instance={result.get('instance')} "
            f"(base_instance={result.get('base_instance')})",
        ]
        if result.get("multi_name"):
            lines.append(f"  multi_name: {result['multi_name']}")
        applied_result = result.get("fields_applied")
        if isinstance(applied_result, dict):
            if applied_result.get("error"):
                lines.append(f"  fields NOT applied: {applied_result['error']}")
            else:
                applied = applied_result.get("applied") or {}
                if applied:
                    lines.append(f"  fields applied: {json.dumps(applied)}")
                unknown = applied_result.get("unknown_fields") or []
                if unknown:
                    lines.append(f"  unknown_fields (ignored): {unknown}")
        return [TextContent(type="text", text="\n".join(lines))]

    async def _handle_get_viewport(self, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            result = self.bridge.call("dev_get_viewport", {}, timeout=15.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(type="text", text=f"get_viewport: {result['error']}")]
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    async def _handle_capture_viewport(self, arguments: Dict[str, Any]) -> List[Any]:
        viewport = arguments.get("viewport", "main")
        if viewport not in ("main", "preview2"):
            return [TextContent(type="text", text="viewport must be 'main' or 'preview2'")]
        max_w = int(arguments.get("max_w", 1400))
        max_h = int(arguments.get("max_h", 1400))
        return_image = bool(arguments.get("return_image", True))

        try:
            vp_result = self.bridge.call("dev_get_viewport", {}, timeout=15.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if vp_result.get("error"):
            return [TextContent(type="text", text=f"capture_viewport: {vp_result['error']}")]

        vp = vp_result.get(viewport)
        if not isinstance(vp, dict) or vp.get("active") is False:
            return [TextContent(
                type="text",
                text=f"capture_viewport: viewport_not_active ({viewport})",
            )]
        region = vp.get("region")
        if not isinstance(region, dict) or not all(k in region for k in ("x", "y", "w", "h")):
            return [TextContent(
                type="text",
                text=(
                    f"capture_viewport: no region available for '{viewport}' "
                    "(no processed pipe yet)"
                ),
            )]
        region = {k: float(region[k]) for k in ("x", "y", "w", "h")}

        # Bind the snapshot to the image it is a picture OF, and re-read the
        # identity after the render (below) so a mid-capture image switch is
        # caught instead of being baked into the snapshot.
        image = self._darkroom_image()
        if not image.get("has_image"):
            return [TextContent(type="text", text=(
                f"capture_viewport: no darkroom image loaded ({image.get('error')}) "
                "-- open an image in darkroom first"
            ))]

        try:
            preview_result = self.bridge.call(
                "dev_preview",
                {"max_w": max_w, "max_h": max_h, "region": region},
                timeout=20.0,
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if preview_result.get("error"):
            return [TextContent(
                type="text",
                text=f"capture_viewport: get_preview failed: {preview_result['error']}",
            )]
        path = preview_result.get("path") or preview_result.get("stale_preview")
        if not path:
            return [TextContent(
                type="text",
                text=f"capture_viewport: no path in preview result: {preview_result}",
            )]
        host_path = _remap_bridge_path(path)
        render_w = int(preview_result.get("width") or 0)
        render_h = int(preview_result.get("height") or 0)

        # The render is only trustworthy if darkroom still holds the same image
        # it did a moment ago -- dev_preview has no image identity of its own.
        image_after = self._darkroom_image()
        if image_after.get("id") != image.get("id"):
            return [TextContent(type="text", text=(
                f"capture_viewport: the darkroom image changed while capturing "
                f"(started on {self._image_label(image)}, ended on "
                f"{self._image_label(image_after)}) -- no snapshot stored, "
                "try again once the view settles"
            ))]

        snapshot_id = self._store_viewport_snapshot({
            "viewport": viewport,
            "region": region,
            "render": {"path": host_path, "width": render_w, "height": render_h},
            "image": image,
        })

        lines = [
            f"capture_viewport('{viewport}'): snapshot_id={snapshot_id}",
            f"  image: id={image.get('id')} filename={image.get('filename')}",
            f"  region: {json.dumps({k: round(region[k], 6) for k in ('x', 'y', 'w', 'h')})}",
            f"  render: {render_w}x{render_h} path={host_path}",
            f"  snapshot_pixels for this snapshot means 0..{render_w} x "
            f"0..{render_h} (the render above), NOT the darktable window size",
            f"  expires in ~{int(self._viewport_snapshot_ttl_s)}s",
        ]
        dl = self._download_url(host_path)
        if dl:
            lines.append(f"  url: {dl} (no auth header needed, token in URL is the credential)")

        out: List[Any] = []
        if return_image:
            img = _inline_image_content(host_path)
            if img is not None:
                out.append(img)
                lines.append("  (inline image: JPEG, downscaled; full-res PNG at path/url above)")
        out.append(TextContent(type="text", text="\n".join(lines)))
        return out

    def _bridge_call_or_error(
        self, method: str, params: Dict[str, Any], timeout: float = 15.0
    ) -> "tuple[Optional[Dict[str, Any]], Optional[TextContent]]":
        """Shared bridge-call + standard plugin/connection error mapping --
        used by set_viewport/restore_viewport, which need several sequential
        bridge round trips (get_viewport, set_viewport, get_viewport again,
        a dev_preview probe) where duplicating the usual
        BridgePluginNotInstalledError/BridgeTimeoutError/BridgeError
        try/except per call (the convention every other handler repeats
        inline) would be excessive. Returns (result, None) on success, or
        (None, TextContent) on a plugin/connection failure. Does NOT inspect
        an {"error": ...} payload the bridge itself returns successfully --
        callers still check that themselves, exactly like every other
        handler's own `if result.get("error")`."""
        try:
            return self.bridge.call(method, params, timeout=timeout), None
        except BridgePluginNotInstalledError:
            return None, TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )
        except BridgeTimeoutError:
            return None, TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )
        except BridgeError as e:
            return None, TextContent(type="text", text=f"Plugin error: {e}")

    async def _handle_set_viewport(self, arguments: Dict[str, Any]) -> List[TextContent]:
        viewport = arguments.get("viewport", "main")
        if viewport not in ("main", "preview2"):
            return [TextContent(type="text", text="viewport must be 'main' or 'preview2'")]

        region_arg = arguments.get("region")
        scale_arg = arguments.get("scale")
        mode_arg = arguments.get("mode")
        given = [v for v in (region_arg, scale_arg, mode_arg) if v is not None]
        if len(given) != 1:
            return [TextContent(type="text", text=(
                "set_viewport: exactly one of 'region', 'scale', or 'mode' is "
                f"required (got {len(given)})"
            ))]

        wait_for_pipe = bool(arguments.get("wait_for_pipe", True))
        timeout_ms = arguments.get("timeout_ms") or 4000
        min_render_px_across = arguments.get("min_render_px_across")

        # --- current state (needed to invert the region formula / keep the
        # current pan for a plain scale change / know if preview2 is open).
        vp_result, err = self._bridge_call_or_error("dev_get_viewport", {})
        if err:
            return [err]
        if vp_result.get("error"):
            return [TextContent(type="text", text=f"set_viewport: {vp_result['error']}")]

        vp = vp_result.get(viewport)
        if not isinstance(vp, dict) or vp.get("active") is False:
            return [TextContent(
                type="text",
                text=f"set_viewport: viewport_not_active ({viewport})",
            )]

        region_now = vp.get("region")
        procw = vp.get("processed_width")
        proch = vp.get("processed_height")
        viewport_w = vp.get("viewport_width")
        viewport_h = vp.get("viewport_height")
        cur_zoom_x = vp.get("zoom_x")
        cur_zoom_y = vp.get("zoom_y")
        if not (isinstance(region_now, dict) and procw and proch and viewport_w and viewport_h):
            return [TextContent(type="text", text=(
                f"set_viewport: no processed pipe yet for '{viewport}' -- open an "
                "image in darkroom first"
            ))]

        requested_region: Optional[Dict[str, float]] = None
        aspect_adjusted = False
        target_scale: Optional[float] = None
        target_zoom_x: Optional[float] = None
        target_zoom_y: Optional[float] = None

        if region_arg is not None:
            if not (isinstance(region_arg, dict)
                    and all(k in region_arg for k in ("x", "y", "w", "h"))):
                return [TextContent(type="text", text="set_viewport: region must be {x,y,w,h}")]
            try:
                rx, ry, rw, rh = (float(region_arg[k]) for k in ("x", "y", "w", "h"))
            except (TypeError, ValueError):
                return [TextContent(type="text", text="set_viewport: region x/y/w/h must be numbers")]
            if rw <= 0 or rh <= 0:
                return [TextContent(type="text", text="set_viewport: region w and h must be > 0")]
            if rx < -1e-6 or ry < -1e-6 or rx + rw > 1.0 + 1e-6 or ry + rh > 1.0 + 1e-6:
                return [TextContent(type="text", text=(
                    "set_viewport: region must be within the full image "
                    "(x,y >= 0, x+w <= 1, y+h <= 1)"
                ))]
            requested_region = {"x": rx, "y": ry, "w": rw, "h": rh}

            # Invert get_viewport's own region formula (region.w =
            # viewport_w/(procw*scale), region.h = viewport_h/(proch*scale) --
            # ONE shared scale, so an aspect-mismatched request is EXPANDED
            # (never cropped, requirement 2) about its own center rather than
            # solved independently per axis.
            k = (viewport_w * proch) / (viewport_h * procw)  # width_shown = k * height_shown
            required_h = max(rh, rw / k)
            width_shown = k * required_h
            if width_shown > rw + 1e-9 or required_h > rh + 1e-9:
                aspect_adjusted = True
            target_scale = viewport_h / (proch * required_h)
            center_x = rx + rw / 2.0
            center_y = ry + rh / 2.0
            target_zoom_x = center_x - 0.5
            target_zoom_y = center_y - 0.5

        elif scale_arg is not None:
            try:
                target_scale = float(scale_arg)
            except (TypeError, ValueError):
                return [TextContent(type="text", text="set_viewport: scale must be a number")]
            # A plain scale change keeps the current pan (zoom under the same
            # center it was already at), like the GUI's own scroll-wheel zoom.
            target_zoom_x = cur_zoom_x
            target_zoom_y = cur_zoom_y

        else:  # mode
            if mode_arg not in ("fit", "fill", "100%", "200%"):
                return [TextContent(type="text", text=(
                    "set_viewport: mode must be one of fit, fill, 100%, 200%"
                ))]
            fit_scale = min(viewport_w / procw, viewport_h / proch)
            fill_scale = max(viewport_w / procw, viewport_h / proch)
            if mode_arg in ("fit", "fill"):
                target_scale = fit_scale if mode_arg == "fit" else fill_scale
                target_zoom_x = 0.0
                target_zoom_y = 0.0
            else:  # "100%" / "200%" -- keep current pan, only change scale.
                target_scale = 1.0 if mode_arg == "100%" else 2.0
                target_zoom_x = cur_zoom_x
                target_zoom_y = cur_zoom_y

        set_result, err = self._bridge_call_or_error(
            "dev_set_viewport",
            {
                "viewport": viewport,
                "zoom_x": target_zoom_x,
                "zoom_y": target_zoom_y,
                "scale": target_scale,
                "wait_for_pipe": wait_for_pipe,
                "timeout_ms": timeout_ms,
            },
            timeout=max(20.0, timeout_ms / 1000.0 + 8.0),
        )
        if err:
            return [err]
        if set_result.get("error"):
            return [TextContent(type="text", text=f"set_viewport: {set_result['error']}")]

        previous = set_result.get("previous") or {}
        clamped_raw = set_result.get("clamped") or []
        # Re-key to the SAME convention set_params' own `clamped` list uses
        # (field/requested/applied/min/max) -- the C binding's own field
        # names (clamped_to/floor/ceiling) are an internal C<->Python detail.
        clamped = [
            {
                "field": c.get("field"),
                "requested": c.get("requested"),
                "applied": c.get("clamped_to"),
                "min": c.get("floor"),
                "max": c.get("ceiling"),
            }
            for c in clamped_raw
        ]
        pipe_ready = bool(set_result.get("pipe_ready", True))
        waited_ms = set_result.get("waited_ms")

        # Re-read: report the ACTUAL resulting state (requirement 4), never
        # the request -- truthful by construction, no duplicated math.
        achieved_region = None
        achieved_scale = None
        vp_result2, err2 = self._bridge_call_or_error("dev_get_viewport", {})
        if not err2 and not vp_result2.get("error"):
            vp2 = vp_result2.get(viewport) or {}
            achieved_region = vp2.get("region")
            achieved_scale = vp2.get("scale")

        # renderable_px: an ACTUAL probe (dev_preview at the achieved region,
        # generous max_w/max_h) rather than a formula -- capture_viewport's
        # own pixel source (dev->preview_pipe) is a FIXED native resolution
        # entirely decoupled from any dev->full/preview2 zoom (see
        # set_viewport_cb's design note, src/lua/develop.c), so computing this
        # from this call's own zoom state would NOT predict what
        # capture_viewport actually returns. A probe measures the real thing.
        renderable_px = None
        if isinstance(achieved_region, dict):
            probe, perr = self._bridge_call_or_error(
                "dev_preview",
                {"max_w": 8192, "max_h": 8192, "region": achieved_region},
            )
            if not perr and not probe.get("error"):
                w, h = probe.get("width"), probe.get("height")
                if w and h:
                    renderable_px = {"w": int(w), "h": int(h)}

        min_render_px_across_satisfied = None
        note = None
        if min_render_px_across is not None and renderable_px is not None:
            try:
                min_px = float(min_render_px_across)
            except (TypeError, ValueError):
                min_px = None
            if min_px is not None:
                min_render_px_across_satisfied = renderable_px["w"] >= min_px
                if not min_render_px_across_satisfied:
                    note = (
                        "min_render_px_across NOT satisfied, and NOT pursued by "
                        "zooming in further: capture_viewport's pixel source "
                        "(dev->preview_pipe) is a fixed native resolution "
                        "decoupled from this viewport's zoom, so shrinking the "
                        "region further would only shrink the crop taken from "
                        "that fixed resolution, not increase it -- zooming in "
                        "would make this WORSE, not better, and would also "
                        "violate 'expand, never crop, the requested region'. "
                        "See CLAUDE.md's 2026-07-31 set_viewport entry."
                    )

        response: Dict[str, Any] = {
            "ok": True,
            "viewport": viewport,
            "achieved": {
                "region": achieved_region,
                "scale": achieved_scale,
                "renderable_px": renderable_px,
            },
            "aspect_adjusted": aspect_adjusted,
            "pipe_ready": pipe_ready,
            "waited_ms": waited_ms,
            "previous": previous,
            "clamped": clamped,
        }
        if requested_region is not None:
            response["requested_region"] = requested_region
        if min_render_px_across_satisfied is not None:
            response["min_render_px_across_satisfied"] = min_render_px_across_satisfied
        if note:
            response["note"] = note

        return [TextContent(type="text", text=json.dumps(response, indent=2))]

    async def _handle_restore_viewport(self, arguments: Dict[str, Any]) -> List[TextContent]:
        viewport = arguments.get("viewport", "main")
        if viewport not in ("main", "preview2"):
            return [TextContent(type="text", text="viewport must be 'main' or 'preview2'")]

        previous = arguments.get("previous")
        if not isinstance(previous, dict) or not all(
            k in previous for k in ("zoom", "closeup", "zoom_x", "zoom_y", "scale")
        ):
            return [TextContent(type="text", text=(
                "restore_viewport: 'previous' must be the object set_viewport "
                "(or get_viewport) returned -- {zoom, closeup, zoom_x, zoom_y, "
                "scale} -- pass it straight through, do not construct it by hand"
            ))]
        wait_for_pipe = bool(arguments.get("wait_for_pipe", True))
        timeout_ms = arguments.get("timeout_ms") or 4000

        result, err = self._bridge_call_or_error(
            "dev_restore_viewport",
            {
                "viewport": viewport,
                "previous": previous,
                "wait_for_pipe": wait_for_pipe,
                "timeout_ms": timeout_ms,
            },
            timeout=max(20.0, timeout_ms / 1000.0 + 8.0),
        )
        if err:
            return [err]
        if result.get("error"):
            return [TextContent(type="text", text=f"restore_viewport: {result['error']}")]

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    # ---- Phase 2 (T2.3): object masks --------------------------------------

    async def _handle_add_path_mask(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        if not op:
            return [TextContent(type="text", text="op is required")]
        points = arguments.get("points")
        if not isinstance(points, list) or len(points) < 3:
            return [TextContent(
                type="text",
                text="points must be an array of at least 3 {x,y} nodes",
            )]
        instance = int(arguments.get("instance", 0))
        opacity = float(arguments.get("opacity", 1.0))
        feather = arguments.get("feather", None)  # None -> bridge/C default (0.02)

        params: Dict[str, Any] = {
            "op": op,
            "instance": instance,
            "points": [{"x": float(pt["x"]), "y": float(pt["y"])} for pt in points],
            "opacity": opacity,
        }
        if feather is not None:
            params["feather"] = float(feather)
        if "smooth" in arguments:
            params["smooth"] = bool(arguments["smooth"])

        try:
            result = self.bridge.call("dev_add_path_mask", params, timeout=15.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(
                type="text",
                text=f"add_path_mask('{op}', instance={instance}): {result['error']}",
            )]
        lines = [
            f"add_path_mask('{op}', instance={instance}): ok=True "
            f"mask_id={result.get('mask_id')} formid={result.get('formid')} "
            f"nodes={result.get('points')} opacity={result.get('opacity')} "
            f"feather={result.get('feather')} smooth={result.get('smooth')}",
        ]
        name = arguments.get("name")
        if name:
            try:
                rename_result = self.bridge.call(
                    "dev_rename_mask", {"mask_id": result.get("formid"), "name": str(name)},
                    timeout=15.0,
                )
                if rename_result.get("error"):
                    lines.append(f"  name NOT set: {rename_result['error']}")
                else:
                    lines.append(f"  name={rename_result.get('name')!r}")
            except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
                lines.append(f"  name NOT set: {e}")
        lines.append("Call get_preview() to see the masked result.")
        return [TextContent(type="text", text="\n".join(lines))]

    async def _handle_retouch_add_shape(self, arguments: Dict[str, Any]) -> List[TextContent]:
        algorithm = arguments.get("algorithm")
        if not algorithm:
            return [TextContent(type="text", text="algorithm is required")]
        target = arguments.get("target")
        if not isinstance(target, dict) or "x" not in target or "y" not in target:
            return [TextContent(type="text", text="target {x,y} is required")]
        source = arguments.get("source")
        if not isinstance(source, dict) or "x" not in source or "y" not in source:
            return [TextContent(type="text", text="source {x,y} is required for heal/clone")]
        radius = arguments.get("radius")
        if radius is None:
            return [TextContent(type="text", text="radius is required")]
        shape_type = arguments.get("shape_type", "circle")
        if shape_type != "circle":
            return [TextContent(type="text", text="only shape_type 'circle' is supported so far")]
        instance = int(arguments.get("instance", 0))

        params: Dict[str, Any] = {
            "op": "retouch",
            "instance": instance,
            "algorithm": algorithm,
            "target": {"x": float(target["x"]), "y": float(target["y"])},
            "source": {"x": float(source["x"]), "y": float(source["y"])},
            "radius": float(radius),
            "feather": float(arguments.get("feather", 0.0)),
            "opacity": float(arguments.get("opacity", 1.0)),
        }
        if arguments.get("wavelet_scale") is not None:
            params["wavelet_scale"] = int(arguments["wavelet_scale"])

        try:
            result = self.bridge.call("dev_retouch_add_shape", params, timeout=15.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(
                type="text",
                text=(
                    f"retouch_add_shape(instance={instance}): "
                    f"{self._explain_darkroom_error(result['error'])}"
                ),
            )]

        lines = [
            f"retouch_add_shape(instance={instance}): ok=True "
            f"formid={result.get('formid')} algorithm={result.get('algorithm')} "
            f"wavelet_scale={result.get('wavelet_scale')} radius={result.get('radius')} "
            f"feather={result.get('feather')} opacity={result.get('opacity')}",
            "Call get_preview() to see the retouched result.",
        ]
        return [TextContent(type="text", text="\n".join(lines))]

    async def _handle_retouch_add_shape_in_viewport(self, arguments: Dict[str, Any]) -> List[Any]:
        snapshot = self._get_viewport_snapshot(arguments.get("snapshot_id"))
        region = snapshot["region"]
        render = snapshot["render"]

        algorithm = arguments.get("algorithm")
        if not algorithm:
            return [TextContent(type="text", text="algorithm is required")]
        target = arguments.get("target")
        source = arguments.get("source")
        radius = arguments.get("radius")
        if radius is None:
            return [TextContent(type="text", text="radius is required")]

        coordinate_space = canonical_space(
            arguments.get("coordinate_space", "snapshot_normalized")
        )
        radius_space = canonical_space(arguments.get("radius_space", coordinate_space))

        # Stage 1: viewport-local -> PROCESSED/DISPLAY-frame normalized
        # (same frame get_viewport()/get_preview() use).
        display_target = viewport_point_to_image(
            region, target, coordinate_space, render["width"], render["height"], label="target"
        )
        display_source = viewport_point_to_image(
            region, source, coordinate_space, render["width"], render["height"], label="source"
        )
        display_radius = viewport_radius_to_image(region, float(radius), radius_space, render["width"])
        feather_local = float(arguments.get("feather", 0.0))
        display_feather = (
            viewport_radius_to_image(region, feather_local, radius_space, render["width"])
            if feather_local > 0
            else 0.0
        )

        # Same-image guard: the snapshot's coordinates only mean something on
        # the image it was captured from, so refuse before touching darktable
        # if darkroom has moved on. Runs after the (free, local) stage-1
        # transform so malformed input still fails without a bridge call.
        guard = self._snapshot_image_guard(snapshot, "retouch_add_shape_in_viewport")
        if guard is not None:
            return guard

        # Stage 2: display-frame -> PIPE-INPUT/mask-frame normalized (the
        # frame retouch_add_shape actually stores coordinates in -- these two
        # frames differ whenever orientation/crop/rotate/lens-correction is
        # active; see _backtransform_to_mask_space's doc comment).
        mask, err = self._backtransform_to_mask_space(
            display_target, display_source, display_radius, display_feather
        )
        if err is not None:
            return err
        full_target = mask["target"]
        full_source = mask["source"]
        full_radius = mask["radius"]
        full_feather = mask["feather"]

        instance = int(arguments.get("instance", 0))
        params: Dict[str, Any] = {
            "op": "retouch",
            "instance": instance,
            "algorithm": algorithm,
            "target": full_target,
            "source": full_source,
            "radius": full_radius,
            "feather": full_feather,
            "opacity": float(arguments.get("opacity", 1.0)),
        }
        if arguments.get("wavelet_scale") is not None:
            params["wavelet_scale"] = int(arguments["wavelet_scale"])

        try:
            result = self.bridge.call("dev_retouch_add_shape", params, timeout=15.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(
                type="text",
                text=(
                    f"retouch_add_shape_in_viewport(instance={instance}): "
                    f"{self._explain_darkroom_error(result['error'])}"
                ),
            )]

        snap_image = snapshot.get("image") or {}
        lines = [
            f"retouch_add_shape_in_viewport(instance={instance}): ok=True "
            f"formid={result.get('formid')} algorithm={result.get('algorithm')} "
            f"wavelet_scale={result.get('wavelet_scale')}",
            f"  image: id={snap_image.get('id')} filename={snap_image.get('filename')} "
            f"(snapshot_id={arguments.get('snapshot_id')})",
            f"  input ({coordinate_space}): target={target} source={source} radius={radius}",
            f"  display-frame: target={display_target} source={display_source} "
            f"radius={display_radius} feather={display_feather}",
            f"  mask-frame (actual write): target={full_target} source={full_source} "
            f"radius={full_radius} feather={full_feather}",
            f"  check placement: retouch_render_overlay(snapshot_id="
            f"'{arguments.get('snapshot_id')}', mode='source_and_target', "
            f"highlight_formid={result.get('formid')})",
        ]

        out: List[Any] = []
        if bool(arguments.get("return_preview", True)):
            preview_img, preview_line = self._render_snapshot_preview(snapshot)
            if preview_img is not None:
                out.append(preview_img)
            lines.append(preview_line)
        out.append(TextContent(type="text", text="\n".join(lines)))
        return out

    async def _handle_retouch_update_shape_in_viewport(self, arguments: Dict[str, Any]) -> List[Any]:
        snapshot = self._get_viewport_snapshot(arguments.get("snapshot_id"))
        region = snapshot["region"]
        render = snapshot["render"]

        formid = arguments.get("formid")
        if formid is None:
            return [TextContent(type="text", text="formid is required")]
        target = arguments.get("target")
        source = arguments.get("source")
        radius = arguments.get("radius")
        if radius is None:
            return [TextContent(type="text", text="radius is required")]

        coordinate_space = canonical_space(
            arguments.get("coordinate_space", "snapshot_normalized")
        )
        radius_space = canonical_space(arguments.get("radius_space", coordinate_space))

        # Stage 1: viewport-local -> PROCESSED/DISPLAY-frame normalized.
        display_target = viewport_point_to_image(
            region, target, coordinate_space, render["width"], render["height"], label="target"
        )
        display_source = viewport_point_to_image(
            region, source, coordinate_space, render["width"], render["height"], label="source"
        )
        display_radius = viewport_radius_to_image(region, float(radius), radius_space, render["width"])
        feather_local = float(arguments.get("feather", 0.0))
        display_feather = (
            viewport_radius_to_image(region, feather_local, radius_space, render["width"])
            if feather_local > 0
            else 0.0
        )

        # Same-image guard (see _snapshot_image_guard).
        guard = self._snapshot_image_guard(snapshot, "retouch_update_shape_in_viewport")
        if guard is not None:
            return guard

        # Stage 2: display-frame -> PIPE-INPUT/mask-frame normalized (see
        # _backtransform_to_mask_space's doc comment).
        mask, err = self._backtransform_to_mask_space(
            display_target, display_source, display_radius, display_feather
        )
        if err is not None:
            return err
        full_target = mask["target"]
        full_source = mask["source"]
        full_radius = mask["radius"]
        full_feather = mask["feather"]

        instance = int(arguments.get("instance", 0))
        params: Dict[str, Any] = {
            "op": "retouch",
            "instance": instance,
            "formid": int(formid),
            "target": full_target,
            "source": full_source,
            "radius": full_radius,
            "feather": full_feather,
        }
        if arguments.get("algorithm"):
            params["algorithm"] = arguments["algorithm"]
        if arguments.get("wavelet_scale") is not None:
            params["wavelet_scale"] = int(arguments["wavelet_scale"])
        if arguments.get("opacity") is not None:
            params["opacity"] = float(arguments["opacity"])

        try:
            result = self.bridge.call("dev_retouch_update_shape", params, timeout=15.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(
                type="text",
                text=(
                    f"retouch_update_shape_in_viewport(formid={formid}): "
                    f"{self._explain_darkroom_error(result['error'])}"
                ),
            )]

        snap_image = snapshot.get("image") or {}
        lines = [
            f"retouch_update_shape_in_viewport(formid={formid}, instance={instance}): ok=True "
            f"algorithm={result.get('algorithm')} wavelet_scale={result.get('wavelet_scale')} "
            f"opacity={result.get('opacity')}",
            f"  image: id={snap_image.get('id')} filename={snap_image.get('filename')} "
            f"(snapshot_id={arguments.get('snapshot_id')})",
            f"  input ({coordinate_space}): target={target} source={source} radius={radius}",
            f"  display-frame: target={display_target} source={display_source} "
            f"radius={display_radius} feather={display_feather}",
            f"  mask-frame (actual write): target={full_target} source={full_source} "
            f"radius={full_radius} feather={full_feather}",
            f"  check placement: retouch_render_overlay(snapshot_id="
            f"'{arguments.get('snapshot_id')}', mode='source_and_target', "
            f"highlight_formid={formid})",
        ]

        out: List[Any] = []
        if bool(arguments.get("return_preview", True)):
            preview_img, preview_line = self._render_snapshot_preview(snapshot)
            if preview_img is not None:
                out.append(preview_img)
            lines.append(preview_line)
        out.append(TextContent(type="text", text="\n".join(lines)))
        return out

    async def _handle_retouch_delete_shape(self, arguments: Dict[str, Any]) -> List[TextContent]:
        formid = arguments.get("formid")
        if formid is None:
            return [TextContent(type="text", text="formid is required")]
        instance = int(arguments.get("instance", 0))
        params = {"op": "retouch", "instance": instance, "formid": int(formid)}

        try:
            result = self.bridge.call("dev_retouch_delete_shape", params, timeout=15.0)
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return [TextContent(
                type="text",
                text=(
                    f"retouch_delete_shape(formid={formid}): "
                    f"{self._explain_darkroom_error(result['error'])}"
                ),
            )]
        return [TextContent(type="text", text=f"Deleted retouch shape {formid}")]

    async def _handle_retouch_delete_shapes(self, arguments: Dict[str, Any]) -> List[TextContent]:
        formids = arguments.get("formids")
        if not isinstance(formids, list) or not formids:
            return [TextContent(type="text", text="formids must be a non-empty array of integers")]
        instance = int(arguments.get("instance", 0))

        deleted: List[int] = []
        failed: List[Dict[str, Any]] = []
        for raw_formid in formids:
            try:
                formid = int(raw_formid)
            except (TypeError, ValueError):
                failed.append({"formid": raw_formid, "error": "not an integer"})
                continue
            params = {"op": "retouch", "instance": instance, "formid": formid}
            try:
                result = self.bridge.call("dev_retouch_delete_shape", params, timeout=15.0)
            except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
                failed.append({"formid": formid, "error": str(e)})
                continue
            if result.get("error"):
                # A delete earlier in this SAME batch can tear down the
                # shared mask group (module->blend_params->mask_id cleared)
                # if it removed the group's last remaining shape -- every
                # subsequent formid in the batch then errors "no shapes
                # group" even though it was already removed along with the
                # group (bugreport 2026-07-25). Verify against
                # retouch_list_shapes before trusting the error: if the
                # formid is genuinely gone, this is an already-deleted
                # no-op, not a real failure.
                if self._retouch_shape_still_exists(instance, formid):
                    failed.append({
                        "formid": formid,
                        "error": self._explain_darkroom_error(result["error"]),
                    })
                else:
                    deleted.append(formid)
            else:
                deleted.append(formid)

        lines = [f"retouch_delete_shapes(instance={instance}): deleted={deleted} failed={len(failed)}"]
        for f in failed:
            lines.append(f"  failed formid={f['formid']}: {f['error']}")
        return [TextContent(type="text", text="\n".join(lines))]

    def _retouch_shape_still_exists(self, instance: int, formid: int) -> bool:
        """Best-effort re-check used by retouch_delete_shapes when a delete
        call errors -- True only if retouch_list_shapes confirms the formid
        is genuinely still present. Any bridge failure here defaults to True
        (assume still present) so a re-check outage never silently reports a
        real failure as success."""
        try:
            result = self.bridge.call(
                "dev_retouch_list_shapes", {"op": "retouch", "instance": instance}, timeout=15.0
            )
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError):
            return True
        if result.get("error"):
            return True
        shapes = result.get("shapes") or []
        return any(s.get("formid") == formid for s in shapes)

    def _list_retouch_shapes(self, instance: int):
        """Shared read of a retouch instance's shapes. Returns
        (result_dict, error_response) -- exactly one is None."""
        params = {"op": "retouch", "instance": instance}
        try:
            result = self.bridge.call("dev_retouch_list_shapes", params, timeout=15.0)
        except BridgePluginNotInstalledError:
            return None, [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return None, [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return None, [TextContent(type="text", text=f"Plugin error: {e}")]

        if result.get("error"):
            return None, [TextContent(
                type="text",
                text=(
                    f"retouch_list_shapes(instance={instance}): "
                    f"{self._explain_darkroom_error(result['error'])}"
                ),
            )]
        return result, None

    async def _handle_retouch_list_shapes(self, arguments: Dict[str, Any]) -> List[TextContent]:
        instance = int(arguments.get("instance", 0))
        result, err = self._list_retouch_shapes(instance)
        if err:
            return err

        shapes = result.get("shapes", [])
        lines = [
            f"retouch(instance={instance}): num_scales={result.get('num_scales')} "
            f"curr_scale={result.get('curr_scale')} "
            f"merge_from_scale={result.get('merge_from_scale')} shapes={len(shapes)}",
        ]
        for s in shapes:
            target = s.get("target") or {}
            source = s.get("source") or {}
            lines.append(
                f"  formid={s.get('formid')} algorithm={s.get('algorithm')} "
                f"shape_type={s.get('shape_type')} "
                f"target=({target.get('x')},{target.get('y')}) "
                f"source=({source.get('x')},{source.get('y')}) "
                f"radius={s.get('radius')} feather={s.get('feather')} "
                f"opacity={s.get('opacity')} "
                f"wavelet_scale={s.get('wavelet_scale')}"
            )
            # Display-frame values (dt.develop.transform_point): the SAME frame
            # capture_viewport/get_preview render in. The mask-frame numbers
            # above are what retouch_add_shape/retouch_update_shape take back,
            # so both are reported -- picking the wrong one silently means
            # comparing a shape against a differently-framed picture of it.
            td = s.get("target_display")
            if isinstance(td, dict):
                sd = s.get("source_display") or {}
                lines.append(
                    f"    display frame: target=({td.get('x')},{td.get('y')}) "
                    f"source=({sd.get('x')},{sd.get('y')}) "
                    f"radius={s.get('radius_display')} "
                    f"feather={s.get('feather_display')}"
                )
        if shapes:
            lines.append(
                "  target/source/radius/feather = mask storage frame (pass these "
                "back to retouch_add_shape/retouch_update_shape). 'display frame' "
                "= the frame get_preview/capture_viewport render in; radius there "
                "is normalized against display WIDTH."
            )
            if not result.get("has_display_frame"):
                lines.append(
                    "  display-frame values unavailable (no processed pipe yet) -- "
                    "call get_preview once and list again"
                )
            lines.append(
                "  Use retouch_render_overlay to SEE these shapes drawn on a "
                "capture_viewport render."
            )
        return [TextContent(type="text", text="\n".join(lines))]

    async def _handle_retouch_render_overlay(self, arguments: Dict[str, Any]) -> List[Any]:
        try:
            snapshot = self._get_viewport_snapshot(arguments.get("snapshot_id"))
        except ViewportCoordinateError as e:
            return [TextContent(type="text", text=f"retouch_render_overlay: {e}")]

        mode = arguments.get("mode", overlay.MODE_ALL_SHAPES)
        instance = int(arguments.get("instance", 0))
        label_shapes = bool(arguments.get("label_shapes", True))
        return_image = bool(arguments.get("return_image", True))
        highlight_formid = arguments.get("highlight_formid")
        if highlight_formid is not None:
            highlight_formid = int(highlight_formid)

        # Read-only, but the geometry is only meaningful for the photo the
        # snapshot is of -- drawing another image's shapes on this render would
        # be a convincing lie, so refuse the same way the write path does.
        expected = snapshot.get("image") or {}
        current = self._darkroom_image()
        if not current.get("has_image"):
            return [TextContent(type="text", text=(
                f"retouch_render_overlay: darkroom has no image open now "
                f"({current.get('error')}); the snapshot is of "
                f"{self._image_label(expected)}. Reopen that image in darkroom."
            ))]
        if expected.get("id") is not None and current.get("id") != expected.get("id"):
            return [TextContent(type="text", text=(
                f"retouch_render_overlay: darkroom image mismatch -- snapshot "
                f"image_id={expected.get('id')} ({expected.get('filename')}), "
                f"active image_id={current.get('id')} ({current.get('filename')}). "
                "The shapes read now belong to the active image, so they must "
                "not be drawn on this snapshot. Reopen the snapshot's image, or "
                "call capture_viewport again for the active one."
            ))]

        result, err = self._list_retouch_shapes(instance)
        if err:
            return err
        shapes = result.get("shapes", [])
        if not shapes:
            return [TextContent(type="text", text=(
                f"retouch_render_overlay: retouch instance {instance} has no "
                "shapes to draw"
            ))]

        render = snapshot["render"]
        render_path = render.get("path")
        out_dir = os.path.dirname(render_path) or tempfile.gettempdir()
        if not os.access(out_dir, os.W_OK):
            out_dir = tempfile.gettempdir()
        out_path = os.path.join(out_dir, f"overlay-{mode}-{uuid.uuid4().hex[:8]}.png")

        try:
            summary = overlay.render_overlay(
                render_path=render_path,
                out_path=out_path,
                shapes=shapes,
                region=snapshot["region"],
                render_width=int(render.get("width") or 0),
                render_height=int(render.get("height") or 0),
                mode=mode,
                highlight_formid=highlight_formid,
                label_shapes=label_shapes,
            )
        except (overlay.OverlayRenderError, ViewportCoordinateError) as e:
            return [TextContent(type="text", text=f"retouch_render_overlay: {e}")]
        except OSError as e:
            return [TextContent(
                type="text", text=f"retouch_render_overlay: could not write overlay: {e}"
            )]

        lines = [
            f"retouch_render_overlay(mode={summary['mode']}, instance={instance}): "
            f"{summary['path']}",
            f"  image: {self._image_label(current)}",
            f"  drawn on snapshot render {summary['render']['width']}x"
            f"{summary['render']['height']} (region "
            f"{json.dumps({k: round(snapshot['region'][k], 6) for k in ('x', 'y', 'w', 'h')})})",
        ]
        if mode == overlay.MODE_MASK_ONLY:
            lines.append(
                "  greyscale = actual mask alpha (white = fully masked), same "
                "quadratic feather falloff darktable applies"
            )
        else:
            lines.append(
                "  red circle+crosshair = target, dashed red = feather edge, "
                "blue = source, yellow line = source->target"
            )
        for g in summary["geometry"]:
            if highlight_formid is not None and g["formid"] != highlight_formid \
                    and mode == overlay.MODE_SOURCE_AND_TARGET:
                continue
            lines.append(
                f"  formid={g['formid']} algorithm={g['algorithm']} "
                f"target_px={g['target_px']} source_px={g['source_px']} "
                f"radius_px={g['radius_px']} feather_px={g['feather_px']} "
                f"opacity={g['opacity']} "
                f"source_target_distance_px={g['source_target_distance_px']}"
            )
        if summary["offscreen"]:
            lines.append(
                f"  partly/fully outside this snapshot: {summary['offscreen']} "
                "(drawn where they fall, but the render does not cover them -- "
                "capture a wider viewport to inspect these)"
            )
        if summary["skipped_no_geometry"]:
            lines.append(
                f"  not drawable: {summary['skipped_no_geometry']} (no "
                "display-frame geometry -- non-circle GUI shape, or no "
                "processed pipe when listed)"
            )
        overlaps = overlay.find_overlaps(summary["geometry"])
        if overlaps:
            lines.append(
                "  OVERLAPPING shapes (a later heal samples the earlier one's "
                "already-healed output): "
                + "; ".join(
                    f"{o['formids']} {o['distance_px']}px apart, reach "
                    f"{o['combined_reach_px']}px"
                    for o in overlaps
                )
            )
        dl = self._download_url(summary["path"])
        if dl:
            lines.append(f"  url: {dl} (no auth header needed, token in URL is the credential)")

        out: List[Any] = []
        if return_image:
            img = _inline_image_content(summary["path"])
            if img is not None:
                out.append(img)
                lines.append("  (inline image: JPEG, downscaled; full-res PNG at path/url above)")
        out.append(TextContent(type="text", text="\n".join(lines)))
        return out

    def _rollback_orphan_instance(self, op: str, instance: int) -> str:
        """Best-effort removal of a module instance created by add_instance
        earlier in the SAME mask_object call, when a later step (add_path_mask
        or set_params) failed. Returns a short human-readable status line to
        append to the error response so the caller knows whether cleanup
        actually happened (ACCEPTANCE T2.3-S2: no orphan instance left).
        """
        try:
            rb = self.bridge.call(
                "dev_remove_last_instance", {"op": op}, timeout=15.0
            )
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            return (
                f"rollback FAILED: could not remove orphan instance "
                f"{op}[{instance}]: {e}. Check list_modules and remove it "
                f"manually if it's still there."
            )
        if not rb.get("ok"):
            return (
                f"rollback FAILED: could not remove orphan instance "
                f"{op}[{instance}]: {rb.get('error')}. Check list_modules "
                f"and remove it manually if it's still there."
            )
        return f"rollback ok: removed orphan instance {op}[{instance}]"

    async def _handle_mask_object(self, arguments: Dict[str, Any]) -> List[Any]:
        op = arguments.get("op")
        adjustment = arguments.get("adjustment")
        if not op:
            return [TextContent(type="text", text="op is required")]
        if not isinstance(adjustment, dict) or not adjustment:
            return [TextContent(type="text", text="adjustment must be a non-empty object")]

        points = arguments.get("points")
        box = arguments.get("box")
        label = arguments.get("label")
        if not points and not box and not label:
            return [TextContent(
                type="text",
                text=(
                    "mask_object needs at least one of points/box/label -- "
                    "look at a get_preview image and pick a point or box on "
                    "the subject first."
                ),
            )]

        opacity = float(arguments.get("opacity", 1.0))
        new_instance = bool(arguments.get("new_instance", True))
        smooth = bool(arguments.get("smooth", True))
        max_nodes = arguments.get("max_nodes")
        max_nodes = int(max_nodes) if max_nodes is not None else None
        min_nodes = arguments.get("min_nodes")
        min_nodes = int(min_nodes) if min_nodes is not None else None

        # (a) full-frame preview -- the sidecar segments THIS render, so its
        # polygon comes back normalized in the same DISPLAY/PROCESSED frame
        # (fine for the overlay drawn on this same image below). It is NOT
        # yet valid for add_path_mask, which stores PIPE-INPUT/mask-frame
        # coordinates -- see step (b2)/_backtransform_polygon_to_mask_space.
        try:
            preview_result = self.bridge.call(
                "dev_preview", {"max_w": 2048, "max_h": 2048}, timeout=20.0
            )
        except BridgePluginNotInstalledError:
            return [TextContent(
                type="text",
                text="darktable-mcp plugin not installed. Run: darktable-mcp install-plugin",
            )]
        except BridgeTimeoutError:
            return [TextContent(
                type="text",
                text="darktable not running, or plugin not loaded. Open darktable and try again.",
            )]
        except BridgeError as e:
            return [TextContent(type="text", text=f"Plugin error: {e}")]

        if preview_result.get("error"):
            return [TextContent(
                type="text",
                text=f"mask_object: get_preview failed: {preview_result['error']}",
            )]
        preview_path = preview_result.get("path") or preview_result.get("stale_preview")
        if not preview_path:
            return [TextContent(
                type="text",
                text=f"mask_object: get_preview returned no path: {preview_result}",
            )]
        host_preview_path = _remap_bridge_path(preview_path)

        # (b) segmentation sidecar -- nothing in darktable has been touched
        # yet, so any failure here is automatically "no partial edit".
        try:
            seg = run_segmentation(
                host_preview_path, points=points, box=box, label=label,
                max_nodes=max_nodes, min_nodes=min_nodes,
            )
        except SegmentationServiceError as e:
            return [TextContent(type="text", text=f"mask_object: {e}")]

        polygon = seg.get("polygon") or []
        if len(polygon) < 3:
            return [TextContent(
                type="text",
                text=(
                    f"mask_object: segmentation returned too few polygon "
                    f"nodes ({len(polygon)}) to form a mask; try a different "
                    f"point/box."
                ),
            )]

        # (b2) frame fix -- convert the sidecar's display-frame polygon into
        # the mask-frame points add_path_mask actually stores (see
        # _backtransform_polygon_to_mask_space's doc comment). Must happen
        # before add_instance so a bad backtransform never leaves an orphan
        # module instance behind.
        mask_polygon, backtransform_err = self._backtransform_polygon_to_mask_space(polygon)
        if backtransform_err is not None:
            return backtransform_err

        # (c) add_instance, if requested -- from here on, ANY failure must
        # roll back this instance before returning (no orphan module).
        instance = 0
        created_instance = False
        if new_instance:
            try:
                inst_result = self.bridge.call("dev_add_instance", {"op": op}, timeout=15.0)
            except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
                return [TextContent(type="text", text=f"mask_object: add_instance failed: {e}")]
            if inst_result.get("error"):
                return [TextContent(
                    type="text",
                    text=f"mask_object: add_instance failed: {inst_result['error']}",
                )]
            instance = int(inst_result.get("instance", 0))
            created_instance = True

        # (d) add_path_mask
        try:
            mask_result = self.bridge.call(
                "dev_add_path_mask",
                {
                    "op": op, "instance": instance, "points": mask_polygon,
                    "opacity": opacity, "smooth": smooth,
                },
                timeout=15.0,
            )
            mask_err = mask_result.get("error")
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            mask_result, mask_err = {}, str(e)

        if mask_err:
            lines = [f"mask_object: add_path_mask failed: {mask_err}"]
            if created_instance:
                lines.append(self._rollback_orphan_instance(op, instance))
            return [TextContent(type="text", text="\n".join(lines))]

        mask_name = arguments.get("name")
        if mask_name:
            try:
                rename_result = self.bridge.call(
                    "dev_rename_mask",
                    {"mask_id": mask_result.get("formid"), "name": str(mask_name)},
                    timeout=15.0,
                )
                name_line = (
                    f"name={rename_result.get('name')!r}" if not rename_result.get("error")
                    else f"name NOT set: {rename_result['error']}"
                )
            except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
                name_line = f"name NOT set: {e}"
        else:
            name_line = (
                f"rename this shape for orientation: "
                f"rename_mask(mask_id={mask_result.get('formid')}, name=...)"
            )

        # (e) set_params -- apply the requested adjustment on the masked instance
        try:
            set_result = self.bridge.call(
                "dev_set_params",
                {"op": op, "instance": instance, "fields": adjustment},
                timeout=15.0,
            )
            set_err = set_result.get("error")
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            set_result, set_err = {}, str(e)

        if set_err:
            lines = [f"mask_object: set_params failed: {set_err}"]
            if created_instance:
                lines.append(self._rollback_orphan_instance(op, instance))
            return [TextContent(type="text", text="\n".join(lines))]

        # (f) fresh preview of the masked result
        try:
            final_preview = self.bridge.call(
                "dev_preview", {"max_w": 1024, "max_h": 1024}, timeout=20.0
            )
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            final_preview = {"error": str(e)}
        final_preview_path = final_preview.get("path") or final_preview.get("stale_preview")

        lines = [
            f"mask_object('{op}', instance={instance}): ok=True",
            f"  segmentation backend: {seg.get('backend')} "
            f"({'precise SAM2' if seg.get('backend') == 'sam2' else 'rough zero-install fallback'})",
            f"  polygon: {len(polygon)} nodes, score={seg.get('score')}, "
            f"bbox_display_frame={json.dumps(seg.get('bbox'))} "
            f"bbox_mask_frame={json.dumps(_polygon_bbox(mask_polygon))}",
            f"  formid={mask_result.get('formid')} mask_id={mask_result.get('mask_id')} "
            f"opacity={mask_result.get('opacity')} ({name_line})",
            f"  applied: {json.dumps(set_result.get('applied') or {})}",
        ]
        resolved_box = seg.get("resolved_box")
        if resolved_box is not None:
            lines.insert(
                3,
                f"  label->box (Grounding DINO): {json.dumps(resolved_box)} "
                "-- verify this is actually the intended subject before trusting the mask",
            )
        clamped = set_result.get("clamped") or []
        for c in clamped:
            lines.append(
                f"  clamped: {c.get('field')} requested {c.get('requested')} -> "
                f"applied {c.get('applied')} (bounds [{c.get('min')}, {c.get('max')}])"
            )
        if final_preview_path:
            lines.append(f"  preview: {_remap_bridge_path(final_preview_path)}")
        else:
            lines.append(f"  preview: unavailable ({final_preview.get('error')})")

        # Return two inline images so the mask is never applied "blind":
        #   1. the mask OVERLAY on the pre-edit render -- shows WHICH region got
        #      selected, so a wrong auto-mask (bright field vs small subject) is
        #      caught immediately and the agent can retry with a tighter prompt;
        #   2. the masked RESULT so the effect is visible in one round-trip.
        out: List[Any] = []
        overlay = _mask_overlay_content(host_preview_path, polygon)
        if overlay is not None:
            out.append(overlay)
            lines.append("  [image 1] mask overlay (red tint = selected region)")
        if final_preview_path:
            result_img = _inline_image_content(_remap_bridge_path(final_preview_path))
            if result_img is not None:
                out.append(result_img)
                lines.append("  [image 2] masked result")
        out.append(TextContent(type="text", text="\n".join(lines)))
        return out

    # ---- Phase 3 (T3.3): raster (matte) masks ------------------------------

    async def _handle_mask_raster(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        adjustment = arguments.get("adjustment")
        if not op:
            return [TextContent(type="text", text="op is required")]
        if not isinstance(adjustment, dict) or not adjustment:
            return [TextContent(type="text", text="adjustment must be a non-empty object")]

        opacity_frac = float(arguments.get("opacity", 1.0))
        new_instance = bool(arguments.get("new_instance", True))

        # Resolve the source file: prefer the path cached by
        # open_image_in_darkroom, else ask darktable directly (covers an image
        # the user opened BY HAND in the GUI) -- no longer a hard error.
        resolved_path = self._resolve_current_image_path()
        if not resolved_path:
            return [TextContent(
                type="text",
                text=(
                    "mask_raster: no image open in darkroom -- open one in the "
                    "GUI or call open_image_in_darkroom (mask_raster acts on "
                    "whichever image is currently open)."
                ),
            )]

        # (a) pick the matting input. MODNet (via PIL) reads ordinary raster
        # formats directly; for anything else (RAW: CR2/NEF/ARW/...) render a
        # full-resolution darkroom preview export first as the best
        # available PIL-readable substitute -- capped by the darkroom
        # preview pipe's own resolution, which is still far more detail than
        # get_preview's usual on-screen sizes since we ask for 4096x4096.
        image_path = resolved_path
        ext = Path(image_path).suffix.lower()
        matte_input_path = image_path
        matte_input_note = f"original image file ({image_path})"
        if ext not in _DIRECT_MATTE_EXTS:
            try:
                pv = self.bridge.call("dev_preview", {"max_w": 4096, "max_h": 4096}, timeout=30.0)
            except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
                return [TextContent(type="text", text=f"mask_raster: full-res preview export failed: {e}")]
            if pv.get("error"):
                return [TextContent(type="text", text=f"mask_raster: full-res preview export failed: {pv['error']}")]
            pv_path = pv.get("path") or pv.get("stale_preview")
            if not pv_path:
                return [TextContent(type="text", text=f"mask_raster: preview export returned no path: {pv}")]
            matte_input_path = _remap_bridge_path(pv_path)
            matte_input_note = (
                f"full-resolution darkroom preview export ({matte_input_path}), "
                f"since '{ext}' isn't directly readable by the matting model"
            )

        # (b) matting sidecar -- nothing in darktable has been touched yet,
        # so any failure here is automatically "no partial edit". (c) write
        # to a UNIQUE path: rasterfile's mask cache is keyed on a hash of its
        # params blob (path+file strings) + image id, NOT file content --
        # reusing a filename across calls would risk serving a stale cached
        # mask if darktable's cache hash happened to collide with a prior
        # run's. A fresh uuid4 filename per call sidesteps that entirely.
        matte_dir = _matte_output_dir()
        matte_filename = f"matte-{uuid.uuid4().hex}.png"
        matte_host_path = str(matte_dir / matte_filename)
        try:
            matte_result = run_matting(matte_input_path, out_path=matte_host_path, backend="modnet")
        except MattingServiceError as e:
            return [TextContent(type="text", text=f"mask_raster: {e}")]

        alpha_host_path = matte_result.get("alpha_path", matte_host_path)
        matte_bridge_path = _host_to_bridge_path(alpha_host_path)
        matte_bridge_dir = str(Path(matte_bridge_path).parent)
        matte_bridge_file = Path(matte_bridge_path).name

        # From here on darktable state changes begin -- ANY failure below
        # must roll back whatever module instance(s) were already created
        # before returning (mirrors mask_object's rollback contract).
        rasterfile_instance = 0
        rf_created = False
        consumer_instance = 0
        consumer_created = False

        def _rollback(lines: List[str]) -> List[TextContent]:
            if consumer_created:
                lines.append(self._rollback_orphan_instance(op, consumer_instance))
            if rf_created:
                lines.append(self._rollback_orphan_instance("rasterfile", rasterfile_instance))
            return [TextContent(type="text", text="\n".join(lines))]

        # (d) add_instance(rasterfile) -- it is NOT ONE_INSTANCE, so a fresh
        # instance per mask_raster call avoids clobbering a rasterfile
        # instance some other call/edit may already be using.
        try:
            rf_inst = self.bridge.call("dev_add_instance", {"op": "rasterfile"}, timeout=15.0)
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            return [TextContent(type="text", text=f"mask_raster: add_instance(rasterfile) failed: {e}")]
        if rf_inst.get("error"):
            return [TextContent(type="text", text=f"mask_raster: add_instance(rasterfile) failed: {rf_inst['error']}")]
        rasterfile_instance = int(rf_inst.get("instance", 0))
        rf_created = True

        try:
            rf_set = self.bridge.call(
                "dev_set_params",
                {"op": "rasterfile", "instance": rasterfile_instance,
                 "fields": {"path": matte_bridge_dir, "file": matte_bridge_file}},
                timeout=15.0,
            )
            rf_set_err = rf_set.get("error")
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            rf_set, rf_set_err = {}, str(e)
        if rf_set_err:
            return _rollback([f"mask_raster: rasterfile set_params failed: {rf_set_err}"])

        try:
            rf_en = self.bridge.call(
                "dev_enable_module",
                {"op": "rasterfile", "instance": rasterfile_instance, "enabled": True},
                timeout=15.0,
            )
            rf_en_err = rf_en.get("error")
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            rf_en, rf_en_err = {}, str(e)
        if rf_en_err:
            return _rollback([f"mask_raster: rasterfile enable failed: {rf_en_err}"])

        # (e) consumer: new instance (default) or instance 0 directly.
        if new_instance:
            try:
                c_inst = self.bridge.call("dev_add_instance", {"op": op}, timeout=15.0)
            except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
                return _rollback([f"mask_raster: add_instance('{op}') failed: {e}"])
            if c_inst.get("error"):
                return _rollback([f"mask_raster: add_instance('{op}') failed: {c_inst['error']}"])
            consumer_instance = int(c_inst.get("instance", 0))
            consumer_created = True

        try:
            set_result = self.bridge.call(
                "dev_set_params",
                {"op": op, "instance": consumer_instance, "fields": adjustment},
                timeout=15.0,
            )
            set_err = set_result.get("error")
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            set_result, set_err = {}, str(e)
        if set_err:
            return _rollback([f"mask_raster: set_params('{op}') failed: {set_err}"])

        # (f) wire: consumer's blend consumes rasterfile's raster mask.
        # set_raster_source's opacity is 0..100 (percent); mask_raster's own
        # 'opacity' input is 0..1 (mask_object convention) -- convert here.
        try:
            wire = self.bridge.call(
                "dev_set_raster_source",
                {"consumer_op": op, "consumer_instance": consumer_instance,
                 "source_op": "rasterfile", "source_instance": rasterfile_instance,
                 "opacity": opacity_frac * 100.0},
                timeout=15.0,
            )
            wire_err = wire.get("error")
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            wire, wire_err = {}, str(e)
        if wire_err:
            return _rollback([f"mask_raster: set_raster_source failed: {wire_err}"])

        try:
            en = self.bridge.call(
                "dev_enable_module",
                {"op": op, "instance": consumer_instance, "enabled": True},
                timeout=15.0,
            )
            en_err = en.get("error")
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            en, en_err = {}, str(e)
        if en_err:
            return _rollback([f"mask_raster: enable_module('{op}') failed: {en_err}"])

        # (g) fresh preview of the matte-masked result.
        try:
            final_preview = self.bridge.call("dev_preview", {"max_w": 1024, "max_h": 1024}, timeout=20.0)
        except (BridgePluginNotInstalledError, BridgeTimeoutError, BridgeError) as e:
            final_preview = {"error": str(e)}
        final_preview_path = final_preview.get("path") or final_preview.get("stale_preview")

        lines = [
            f"mask_raster('{op}', instance={consumer_instance}): ok=True",
            f"  matting input: {matte_input_note}",
            f"  matte backend: {matte_result.get('backend')} "
            f"(alpha min/max/mean = {matte_result.get('alpha_min'):.4f}/"
            f"{matte_result.get('alpha_max'):.4f}/{matte_result.get('alpha_mean'):.4f}, "
            f"size {matte_result.get('size')})",
            f"  matte_path: {alpha_host_path}",
            f"  rasterfile_instance: {rasterfile_instance}",
            f"  consumer_instance: {consumer_instance}",
            f"  opacity: {opacity_frac}",
            f"  applied: {json.dumps(set_result.get('applied') or {})}",
        ]
        clamped = set_result.get("clamped") or []
        for c in clamped:
            lines.append(
                f"  clamped: {c.get('field')} requested {c.get('requested')} -> "
                f"applied {c.get('applied')} (bounds [{c.get('min')}, {c.get('max')}])"
            )
        if final_preview_path:
            lines.append(f"  preview: {_remap_bridge_path(final_preview_path)}")
        else:
            lines.append(f"  preview: unavailable ({final_preview.get('error')})")
        return [TextContent(type="text", text="\n".join(lines))]

    async def _handle_apply_ratings_batch(self, arguments: Dict[str, Any]) -> List[TextContent]:
        source_dir = arguments.get("source_dir")
        ratings = arguments.get("ratings") or {}
        if not source_dir:
            return [TextContent(type="text", text="source_dir is required")]
        if not ratings:
            return [TextContent(type="text", text="ratings must be a non-empty map")]
        result = apply_ratings_batch(
            source_dir=source_dir,
            ratings=ratings,
            log=bool(arguments.get("log", True)),
        )
        return [TextContent(type="text", text=format_ratings_summary(result))]

    async def start(self) -> None:
        """Run the MCP server over stdio."""
        async with stdio_server() as (read_stream, write_stream):
            await self.app.run(
                read_stream,
                write_stream,
                self.app.create_initialization_options(),
            )

    async def run(self) -> None:
        """Run the MCP server using stdio transport."""
        logger.info("Starting Darktable MCP Server (stdio)")
        await self.start()

    async def run_http(
        self,
        host: str = "127.0.0.1",
        port: int = 8787,
        path: str = "/mcp",
        bearer_token: Optional[str] = None,
        json_response: bool = False,
    ) -> None:
        """Run the MCP server over Streamable HTTP (for remote/proxied access).

        Wraps the same ``self.app`` the stdio transport uses in a Starlette ASGI
        app driven by ``StreamableHTTPSessionManager`` and served by uvicorn, so
        the tool set is identical across transports. A static Bearer check is
        enforced by a pure-ASGI middleware (deliberately NOT a Starlette
        ``BaseHTTPMiddleware``, which would buffer the streamed response body and
        break the server->client SSE stream). Intended to sit behind an HTTPS
        reverse proxy (e.g. nginx) -- do not expose plain HTTP to the internet.
        """
        import contextlib

        import uvicorn
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.responses import FileResponse, PlainTextResponse
        from starlette.routing import Mount, Route

        session_manager = StreamableHTTPSessionManager(
            app=self.app,
            json_response=json_response,
            stateless=False,
        )

        async def handle_mcp(scope: Any, receive: Any, send: Any) -> None:
            await session_manager.handle_request(scope, receive, send)

        async def serve_file(request: Any) -> Any:
            """Stream a registered file (preview/export) by its download token.

            Only paths the server put in the registry are reachable, so there is
            no path-traversal surface. FileResponse streams, so a 120MB TIFF is
            served without buffering it into memory."""
            token = request.path_params.get("token", "")
            fpath = self._download_registry.get(token)
            if not fpath or not os.path.isfile(fpath):
                return PlainTextResponse("Not Found", status_code=404)
            return FileResponse(fpath)

        @contextlib.asynccontextmanager
        async def lifespan(_app: Any):
            async with session_manager.run():
                logger.info(
                    "Darktable MCP Server (streamable HTTP) listening on "
                    "http://%s:%s%s",
                    host,
                    port,
                    path,
                )
                yield

        # Mount at root, not at `path`: StreamableHTTPSessionManager routes on
        # HTTP method + session header, not URL path, so a root mount serves the
        # endpoint at `/mcp`, `/mcp/`, and any subpath with no 307 slash-redirect
        # (which some MCP clients mishandle) and no 404 on the bare path. `path`
        # stays the advertised external route the reverse proxy forwards.
        # The download route MUST come before the catch-all root mount
        # (Starlette matches in order); it lives under `path` so the existing
        # nginx `location /mcp` forwards it -- no extra vhost rule needed.
        asgi_app: Any = Starlette(
            routes=[
                Route(
                    path.rstrip("/") + "/files/{token}",
                    serve_file,
                    methods=["GET"],
                ),
                Mount("", app=handle_mcp),
            ],
            lifespan=lifespan,
        )

        if bearer_token:
            asgi_app = _BearerAuthMiddleware(
                asgi_app,
                bearer_token,
                bypass_prefix=path.rstrip("/") + "/files/",
            )
        else:
            logger.warning(
                "No bearer token set (DTMCP_BEARER_TOKEN empty) -- HTTP endpoint "
                "is UNAUTHENTICATED. Anyone who can reach it controls darktable "
                "and the local disk."
            )

        config = uvicorn.Config(asgi_app, host=host, port=port, log_level="info")
        await uvicorn.Server(config).serve()
