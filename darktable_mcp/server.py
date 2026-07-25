"""Main MCP server for darktable integration."""

import base64
import hmac
import json
import logging
import os
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
from .tools.segmentation_tools import run_segmentation
from .utils.errors import DarktableMCPError, MattingServiceError, SegmentationServiceError

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
                    "Write an AI-generated assessment or note for a photo, stored "
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
                    "Extract auto-rotated JPEG previews from a directory of "
                    "raw files (NEF/CR2/ARW/DNG/etc) for vision-based rating. "
                    "Each preview is rotated upright via EXIF orientation and "
                    "resized to max_dim (default 1024). A smaller thumb_dim "
                    "(default 384) is also written for token-efficient "
                    "first-pass culling. Returns a list of items with preview "
                    "paths plus an EXIF summary (ISO, shutter, focal, "
                    "aperture, datetime) per file."
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
                    "from view_photos drops in directly."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "photo_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Absolute paths to source images",
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
                    "Call list_modules() afterward to see both instances."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "op": {
                            "type": "string",
                            "description": "Module operation name, e.g. 'exposure' (see list_modules)",
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
            # ---- Phase 2 (T2.3): object masks -------------------------------
            Tool(
                name="add_path_mask",
                description=(
                    "LOW-LEVEL: attach a drawn polygon path mask to a module "
                    "instance so its effect only applies inside that shape. "
                    "You usually want mask_object instead — it does the "
                    "vision-pick -> segment -> mask -> edit sequence for you "
                    "in one call. Use add_path_mask directly only if you "
                    "already have a normalized polygon from somewhere (e.g. "
                    "you already called mask_object and want to attach the "
                    "same polygon to a different module/instance). points "
                    "must be an array of >=3 {x,y} nodes normalized 0..1 "
                    "against the full image frame."
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
                    },
                    "required": ["op", "points"],
                },
            ),
            Tool(
                name="retouch_add_shape",
                description=(
                    "Create a local HEAL or CLONE circle shape on the retouch "
                    "module — the module's actual local-editing surface, "
                    "distinct from add_path_mask's generic 'restrict this "
                    "module's blend to a region'. Fixes dust/blemishes/small "
                    "distractions by sampling a source region onto a target "
                    "region, optionally on a specific wavelet scale (skin "
                    "texture vs base tones vs residual detail). Only "
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
                            "description": "Shape center, normalized 0..1 against the full image frame",
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
                            "description": "Circle radius, normalized 0..1 against the full image frame",
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
                    "image, picked visually — e.g. 'brighten her face', "
                    "'darken the sky'. YOU (the vision model) look at a "
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
                                "One or more prompt points on the subject, "
                                "normalized 0..1 against the full frame. "
                                "Usually one point on the object center is "
                                "enough for SAM2."
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
                                "Free-text hint (e.g. 'face'), best-effort "
                                "only — SAM2 has no text grounding, so this "
                                "does nothing useful unless points/box are "
                                "also given. Always prefer picking points/"
                                "box yourself from the preview image."
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
                    },
                    "required": ["op", "adjustment"],
                },
            ),
            # ---- Phase 3 (T3.3): raster (matte) masks ----------------------
            Tool(
                name="mask_raster",
                description=(
                    "Apply a local edit gated by a SOFT-EDGED alpha matte "
                    "(hair, fur, fine wispy detail) instead of a hard drawn "
                    "polygon -- use this instead of mask_object whenever the "
                    "subject has fine edges a path mask would jag up (a "
                    "hairline, flyaway strands, fur, motion blur, glass, "
                    "smoke). It acts on whichever image is currently open in "
                    "darkroom (call open_image_in_darkroom first) with NO "
                    "point/box picking needed -- unlike mask_object, the "
                    "MODNet matting model is a dense, whole-image, "
                    "unprompted portrait matte, so there is nothing to pick. "
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
            "get_preview": self._handle_get_preview,
            "enable_module": self._handle_enable_module,
            "add_instance": self._handle_add_instance,
            "get_viewport": self._handle_get_viewport,
            "add_path_mask": self._handle_add_path_mask,
            "retouch_add_shape": self._handle_retouch_add_shape,
            "retouch_delete_shape": self._handle_retouch_delete_shape,
            "retouch_list_shapes": self._handle_retouch_list_shapes,
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

        if not output_path:
            return [TextContent(type="text", text="output_path is required")]
        if not photo_ids:
            return [
                TextContent(
                    type="text",
                    text="photo_ids must contain at least one path",
                )
            ]

        input_files = [Path(p) for p in photo_ids]
        out_dir = Path(output_path)
        results = self.cli.batch_export(
            input_files=input_files,
            output_dir=out_dir,
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
                f"filename={info.get('filename')} path={host_path}"
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
        if result.get("width") and result.get("height"):
            lines.append(f"size: {result['width']}x{result['height']}")
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

    async def _handle_add_instance(self, arguments: Dict[str, Any]) -> List[TextContent]:
        op = arguments.get("op")
        if not op:
            return [TextContent(type="text", text="op is required")]
        try:
            result = self.bridge.call("dev_add_instance", {"op": op}, timeout=15.0)
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
            "Call get_preview() to see the masked result.",
        ]
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
                text=f"retouch_add_shape(instance={instance}): {result['error']}",
            )]

        lines = [
            f"retouch_add_shape(instance={instance}): ok=True "
            f"formid={result.get('formid')} algorithm={result.get('algorithm')} "
            f"wavelet_scale={result.get('wavelet_scale')} radius={result.get('radius')} "
            f"feather={result.get('feather')} opacity={result.get('opacity')}",
            "Call get_preview() to see the retouched result.",
        ]
        return [TextContent(type="text", text="\n".join(lines))]

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
                text=f"retouch_delete_shape(formid={formid}): {result['error']}",
            )]
        return [TextContent(type="text", text=f"Deleted retouch shape {formid}")]

    async def _handle_retouch_list_shapes(self, arguments: Dict[str, Any]) -> List[TextContent]:
        instance = int(arguments.get("instance", 0))
        params = {"op": "retouch", "instance": instance}

        try:
            result = self.bridge.call("dev_retouch_list_shapes", params, timeout=15.0)
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
                text=f"retouch_list_shapes(instance={instance}): {result['error']}",
            )]

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
                f"wavelet_scale={s.get('wavelet_scale')}"
            )
        return [TextContent(type="text", text="\n".join(lines))]

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

        # (a) full-frame preview -- the sidecar's coords line up 1:1 with
        # this render, so the polygon it returns is directly valid for
        # add_path_mask without any rescaling.
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
                host_preview_path, points=points, box=box, label=label
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
                {"op": op, "instance": instance, "points": polygon, "opacity": opacity},
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
            f"bbox={json.dumps(seg.get('bbox'))}",
            f"  mask_id={mask_result.get('mask_id')} opacity={mask_result.get('opacity')}",
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
