"""Acceptance tests for the honesty-pass iteration.

Each test pins one piece of the desired final state. They are red at the
start of the iteration and turn green incrementally as deletion tasks
land. See docs/superpowers/specs/2026-04-27-honesty-pass-design.md.

These tests stay in the suite as a regression guard after the iteration
completes — do not delete this file when the tests go green.
"""

import importlib

import pytest

from darktable_mcp.server import DarktableMCPServer

# Updated 2026-04-27 (iteration 2): added view_photos + rate_photos restored via IPC bridge.
EXPECTED_TOOLS = {
    "import_from_camera",
    "export_images",
    "extract_previews",
    "apply_ratings_batch",
    "open_in_darktable",
    "view_photos",
    # Contact-sheet spec: grid-of-thumbnails visual culling over the current
    # collection (see docs/contact-sheet spec).
    "get_contact_sheet",
    "rate_photos",
    # DAM extension: tagging + tag-backed collections (darktable's Lua API
    # has no separate "collection" object, so collections == tags).
    "tag_photo",
    # AI-written per-photo assessment/notes, stored in the standard
    # dc:description xmp field (image.description in darktable's Lua API).
    "set_photo_note",
    "get_photo_note",
    "list_collections",
    "list_photos_in_collection",
    "import_batch",
    "list_styles",
    "apply_preset",
    # T1.6: Phase 1 scalar darkroom-editing tools (see PLAN.md §5 T1.6).
    "open_image_in_darkroom",
    "navigate_photo",
    "get_current_image",
    "list_modules",
    "get_params",
    "set_params",
    "get_preview",
    # T1.7: enable/instance/viewport wiring (see PLAN.md §5 T1.7).
    "enable_module",
    "add_instance",
    "get_viewport",
    # T2.3: object masks (see PLAN.md §5 T2.2/T2.3).
    "add_path_mask",
    "mask_object",
    # T3.3: raster (matte) masks (see PLAN.md §5 T3.3).
    "mask_raster",
    # Retouch Phase 1: local heal/clone circle shapes (rt_forms + wavelet
    # scale), see PLAN.md's retouch section.
    "retouch_add_shape",
    "retouch_delete_shape",
    "retouch_list_shapes",
}


def test_server_registers_exactly_the_surviving_tools():
    """No broken or stubbed library tools advertised."""
    server = DarktableMCPServer()
    assert set(server.list_tools()) == EXPECTED_TOOLS


def test_lua_executor_module_is_removed():
    with pytest.raises(ImportError):
        importlib.import_module("darktable_mcp.darktable.lua_executor")


def test_library_detector_module_is_removed():
    with pytest.raises(ImportError):
        importlib.import_module("darktable_mcp.darktable.library_detector")


def test_photo_tools_module_is_removed():
    """Renamed to camera_tools.py."""
    with pytest.raises(ImportError):
        importlib.import_module("darktable_mcp.tools.photo_tools")


def test_camera_tools_module_exposes_camera_tools_class():
    mod = importlib.import_module("darktable_mcp.tools.camera_tools")
    assert hasattr(mod, "CameraTools")
    assert not hasattr(mod, "PhotoTools")
