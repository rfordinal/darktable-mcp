"""Tests for the main MCP server."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest

from darktable_mcp.server import DarktableMCPServer


class TestDarktableMCPServer:
    """Test cases for DarktableMCPServer."""

    def test_server_initialization(self):
        server = DarktableMCPServer()
        assert server is not None
        assert hasattr(server, "app")

    def test_server_has_required_tools(self):
        server = DarktableMCPServer()
        expected_tools = {
            "import_from_camera",
            "export_images",
            "extract_previews",
            "apply_ratings_batch",
            "open_in_darktable",
            "view_photos",
            "get_contact_sheet",
            "rate_photos",
            "tag_photo",
            "set_photo_note",
            "get_photo_note",
            "list_collections",
            "list_photos_in_collection",
            "import_batch",
            "list_styles",
            "apply_preset",
            "open_image_in_darkroom",
            "navigate_photo",
            "get_current_image",
            "list_modules",
            "get_params",
            "set_params",
            "get_preview",
            "enable_module",
            "add_instance",
            "get_viewport",
            "add_path_mask",
            "retouch_add_shape",
            "retouch_delete_shape",
            "retouch_list_shapes",
            "mask_object",
            "mask_raster",
        }
        assert set(server.list_tools()) == expected_tools

    @pytest.mark.asyncio
    async def test_server_can_start(self):
        server = DarktableMCPServer()

        @asynccontextmanager
        async def fake_stdio():
            yield (AsyncMock(), AsyncMock())

        with patch("darktable_mcp.server.stdio_server", fake_stdio), patch.object(
            server.app, "run", new=AsyncMock(return_value=None)
        ) as mock_run:
            await server.start()
            mock_run.assert_called_once()


@pytest.mark.asyncio
async def test_import_from_camera_handler():
    server = DarktableMCPServer()
    mock_tools = Mock()
    mock_tools.import_from_camera.return_value = (
        "Copied 5 file(s) from Nikon DSC D800E (usb:002,002)\n"
        "Destination: /tmp/import-2026-04-26"
    )
    server.camera_tools = mock_tools

    result = await server._handle_import_from_camera({"destination": "/tmp/import-2026-04-26"})

    assert len(result) == 1
    assert "Copied 5 file(s)" in result[0].text
    assert "Nikon DSC D800E" in result[0].text
    mock_tools.import_from_camera.assert_called_once_with({"destination": "/tmp/import-2026-04-26"})


def test_server_registers_import_from_camera_tool():
    server = DarktableMCPServer()
    tool_names = [t.name for t in server._tool_definitions()]
    assert "import_from_camera" in tool_names


@pytest.mark.asyncio
async def test_handle_view_photos_returns_formatted_list():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = [
        {"id": "1", "filename": "a.NEF", "path": "/photos/a.NEF", "rating": 5},
        {"id": "2", "filename": "b.NEF", "path": "/photos/b.NEF", "rating": 4},
    ]
    result = await server._handle_view_photos({"filter": "", "limit": 10})
    assert len(result) == 1
    text = result[0].text
    assert "a.NEF" in text
    assert "b.NEF" in text
    # The absolute file path must be in the formatted output so the agent
    # can pass it straight to export_images. Otherwise view_photos and
    # export_images don't compose.
    assert "/photos/a.NEF" in text
    assert "/photos/b.NEF" in text
    server.bridge.call.assert_called_once_with(
        "view_photos", {"filter": "", "limit": 10}
    )


@pytest.mark.asyncio
async def test_handle_view_photos_no_results_message():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = []
    result = await server._handle_view_photos({})
    assert "No photos" in result[0].text


@pytest.mark.asyncio
async def test_handle_view_photos_friendly_message_when_plugin_missing():
    from darktable_mcp.bridge.client import BridgePluginNotInstalledError

    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.side_effect = BridgePluginNotInstalledError("missing")
    result = await server._handle_view_photos({})
    assert "install-plugin" in result[0].text


@pytest.mark.asyncio
async def test_handle_view_photos_friendly_message_when_dt_not_running():
    from darktable_mcp.bridge.client import BridgeTimeoutError

    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.side_effect = BridgeTimeoutError("timeout")
    result = await server._handle_view_photos({})
    assert "darktable" in result[0].text.lower()
    assert "open" in result[0].text.lower() or "running" in result[0].text.lower()


@pytest.mark.asyncio
async def test_handle_rate_photos_returns_count():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {"updated": 3}
    result = await server._handle_rate_photos({"photo_ids": ["1", "2", "3"], "rating": 4})
    assert "3" in result[0].text
    assert "4" in result[0].text
    server.bridge.call.assert_called_once_with(
        "rate_photos", {"photo_ids": ["1", "2", "3"], "rating": 4}
    )


@pytest.mark.asyncio
async def test_handle_set_photo_note_saved():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {"updated": 1}
    result = await server._handle_set_photo_note({"photo_id": "1", "note": "sharp, good crop"})
    assert "saved" in result[0].text.lower()
    assert "1" in result[0].text
    server.bridge.call.assert_called_once_with(
        "set_photo_note", {"photo_id": "1", "note": "sharp, good crop"}
    )


@pytest.mark.asyncio
async def test_handle_set_photo_note_missing_photo():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {"updated": 0}
    result = await server._handle_set_photo_note({"photo_id": "999", "note": "x"})
    assert "not found" in result[0].text.lower()


@pytest.mark.asyncio
async def test_handle_get_photo_note_returns_text():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {"found": True, "note": "sharp, good crop"}
    result = await server._handle_get_photo_note({"photo_id": "1"})
    assert result[0].text == "sharp, good crop"
    server.bridge.call.assert_called_once_with("get_photo_note", {"photo_id": "1"})


@pytest.mark.asyncio
async def test_handle_get_photo_note_missing_photo():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {"found": False}
    result = await server._handle_get_photo_note({"photo_id": "999"})
    assert "not found" in result[0].text.lower()


@pytest.mark.asyncio
async def test_handle_import_batch_returns_count():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {"imported": 12, "source_path": "/path/foo"}
    result = await server._handle_import_batch({"source_path": "/path/foo"})
    assert "Imported 12" in result[0].text
    assert "/path/foo" in result[0].text
    server.bridge.call.assert_called_once_with(
        "import_batch", {"source_path": "/path/foo"}
    )


@pytest.mark.asyncio
async def test_handle_import_batch_friendly_error_when_plugin_missing():
    from darktable_mcp.bridge.client import BridgePluginNotInstalledError

    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.side_effect = BridgePluginNotInstalledError("missing")
    result = await server._handle_import_batch({"source_path": "/x"})
    assert "install-plugin" in result[0].text


@pytest.mark.asyncio
async def test_handle_import_batch_friendly_error_when_dt_not_running():
    from darktable_mcp.bridge.client import BridgeTimeoutError

    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.side_effect = BridgeTimeoutError("timeout")
    result = await server._handle_import_batch({"source_path": "/x"})
    assert "darktable" in result[0].text.lower()


@pytest.mark.asyncio
async def test_handle_list_styles_returns_count():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {
        "styles": [{"name": "alpha", "description": "a"}, {"name": "beta", "description": "b"}],
        "count": 2,
    }
    result = await server._handle_list_styles({})
    text = result[0].text
    assert "2 styles installed" in text
    assert "alpha" in text
    assert "beta" in text


@pytest.mark.asyncio
async def test_handle_list_styles_empty():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {"styles": [], "count": 0}
    result = await server._handle_list_styles({})
    assert "No styles" in result[0].text


@pytest.mark.asyncio
async def test_handle_list_styles_truncates_at_50():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {
        "styles": [{"name": f"s{i}", "description": ""} for i in range(75)],
        "count": 75,
    }
    result = await server._handle_list_styles({})
    text = result[0].text
    assert "75 styles installed" in text
    assert "and 25 more" in text


@pytest.mark.asyncio
async def test_handle_apply_preset_returns_applied_count():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {"applied": 3, "missed": [], "preset_name": "myStyle"}
    result = await server._handle_apply_preset({
        "photo_ids": ["1", "2", "3"], "preset_name": "myStyle",
    })
    text = result[0].text
    assert "myStyle" in text
    assert "3 photo" in text


@pytest.mark.asyncio
async def test_handle_apply_preset_reports_missed():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {
        "applied": 1, "missed": ["999"], "preset_name": "myStyle",
    }
    result = await server._handle_apply_preset({
        "photo_ids": ["1", "999"], "preset_name": "myStyle",
    })
    text = result[0].text
    assert "999" in text
    assert "Missed" in text


@pytest.mark.asyncio
async def test_handle_apply_preset_friendly_error_when_dt_not_running():
    from darktable_mcp.bridge.client import BridgeTimeoutError

    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.side_effect = BridgeTimeoutError("timeout")
    result = await server._handle_apply_preset({
        "photo_ids": ["1"], "preset_name": "x",
    })
    assert "darktable" in result[0].text.lower()


@pytest.mark.asyncio
async def test_handle_export_images_writes_side_file_and_short_summary(tmp_path):
    """`export_images` used to dump the full per-file map inline, blowing
    Claude's token budget at 400+ files. The handler now writes per-file
    results to a JSONL side file and returns only counts + the path."""
    import json

    server = DarktableMCPServer()
    fake_results = {
        "/in/A.NEF": f"Exported to {tmp_path}/A.jpg",
        "/in/B.NEF": f"Exported to {tmp_path}/B.jpg",
        "/in/C.NEF": "Error: Failed to export image: Export failed: Boom",
    }
    server._cli = Mock()
    server._cli.batch_export.return_value = fake_results

    result = await server._handle_export_images({
        "photo_ids": ["/in/A.NEF", "/in/B.NEF", "/in/C.NEF"],
        "output_path": str(tmp_path),
        "format": "jpeg",
        "quality": 95,
    })
    text = result[0].text

    # Summary stays compact: counts + side file pointer + first error.
    assert "exported: 2" in text
    assert "failed: 1" in text
    assert ".export_images.jsonl" in text
    assert "Boom" in text  # first error gets a short snippet
    # Crucially, the per-file output map is NOT inlined.
    assert "/in/A.NEF" not in text
    assert "/in/B.NEF" not in text

    # Side file holds the full record, one JSON per line.
    side = tmp_path / ".export_images.jsonl"
    assert side.exists()
    lines = [json.loads(l) for l in side.read_text().splitlines()]
    assert len(lines) == 3
    assert {l["input"] for l in lines} == set(fake_results)


@pytest.mark.asyncio
async def test_handle_export_images_validates_required_args():
    server = DarktableMCPServer()
    r = await server._handle_export_images({"photo_ids": ["/x"]})
    assert "output_path" in r[0].text
    r = await server._handle_export_images({"output_path": "/tmp/x"})
    assert "photo_ids" in r[0].text


@pytest.mark.asyncio
async def test_handle_retouch_add_shape_success():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {
        "ok": True, "formid": 42, "algorithm": "heal", "wavelet_scale": 2,
        "radius": 0.02, "feather": 0.1, "opacity": 1.0,
    }
    result = await server._handle_retouch_add_shape({
        "algorithm": "heal",
        "target": {"x": 0.4, "y": 0.3},
        "source": {"x": 0.35, "y": 0.3},
        "radius": 0.02,
        "wavelet_scale": 2,
    })
    assert "formid=42" in result[0].text
    server.bridge.call.assert_called_once_with(
        "dev_retouch_add_shape",
        {
            "op": "retouch",
            "instance": 0,
            "algorithm": "heal",
            "target": {"x": 0.4, "y": 0.3},
            "source": {"x": 0.35, "y": 0.3},
            "radius": 0.02,
            "feather": 0.0,
            "opacity": 1.0,
            "wavelet_scale": 2,
        },
        timeout=15.0,
    )


@pytest.mark.asyncio
async def test_handle_retouch_add_shape_requires_source():
    server = DarktableMCPServer()
    server.bridge = Mock()
    result = await server._handle_retouch_add_shape({
        "algorithm": "heal",
        "target": {"x": 0.4, "y": 0.3},
        "radius": 0.02,
    })
    assert "source" in result[0].text
    server.bridge.call.assert_not_called()


@pytest.mark.asyncio
async def test_handle_retouch_add_shape_rejects_non_circle():
    server = DarktableMCPServer()
    server.bridge = Mock()
    result = await server._handle_retouch_add_shape({
        "algorithm": "heal",
        "shape_type": "path",
        "target": {"x": 0.4, "y": 0.3},
        "source": {"x": 0.35, "y": 0.3},
        "radius": 0.02,
    })
    assert "circle" in result[0].text
    server.bridge.call.assert_not_called()


@pytest.mark.asyncio
async def test_handle_retouch_add_shape_propagates_c_error():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {"error": "300 shape limit reached"}
    result = await server._handle_retouch_add_shape({
        "algorithm": "clone",
        "target": {"x": 0.4, "y": 0.3},
        "source": {"x": 0.35, "y": 0.3},
        "radius": 0.02,
    })
    assert "300 shape limit reached" in result[0].text


@pytest.mark.asyncio
async def test_handle_retouch_delete_shape_success():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {"ok": True}
    result = await server._handle_retouch_delete_shape({"formid": 42})
    assert "42" in result[0].text
    server.bridge.call.assert_called_once_with(
        "dev_retouch_delete_shape",
        {"op": "retouch", "instance": 0, "formid": 42},
        timeout=15.0,
    )


@pytest.mark.asyncio
async def test_handle_retouch_delete_shape_requires_formid():
    server = DarktableMCPServer()
    server.bridge = Mock()
    result = await server._handle_retouch_delete_shape({})
    assert "formid" in result[0].text
    server.bridge.call.assert_not_called()


@pytest.mark.asyncio
async def test_handle_retouch_list_shapes_formats_shapes():
    server = DarktableMCPServer()
    server.bridge = Mock()
    server.bridge.call.return_value = {
        "module": "retouch", "instance": 0,
        "num_scales": 4, "curr_scale": 1, "merge_from_scale": 0,
        "shapes": [
            {
                "formid": 42, "algorithm": "heal", "shape_type": "circle",
                "target": {"x": 0.4, "y": 0.3}, "source": {"x": 0.35, "y": 0.3},
                "radius": 0.02, "feather": 0.1, "wavelet_scale": 2,
            },
        ],
    }
    result = await server._handle_retouch_list_shapes({})
    text = result[0].text
    assert "shapes=1" in text
    assert "formid=42" in text
    assert "algorithm=heal" in text
    server.bridge.call.assert_called_once_with(
        "dev_retouch_list_shapes", {"op": "retouch", "instance": 0}, timeout=15.0
    )


COLLECTION_FIXTURE = [
    {"id": "1", "filename": "a.NEF", "path": "/photos/a.NEF", "rating": 0, "capture_time": "", "selected": False},
    {"id": "2", "filename": "b.NEF", "path": "/photos/b.NEF", "rating": 0, "capture_time": "", "selected": False},
    {"id": "3", "filename": "c.NEF", "path": "/photos/c.NEF", "rating": 0, "capture_time": "", "selected": False},
]


def _bridge_router(responses):
    def _call(method, params=None, timeout=None):
        return responses[method]
    bridge = Mock()
    bridge.call = Mock(side_effect=_call)
    return bridge


@pytest.mark.asyncio
async def test_navigate_photo_next_opens_neighbor():
    server = DarktableMCPServer()
    server.bridge = _bridge_router({
        "dev_current_image": {"has_image": True, "id": 1, "path": "/photos/a.NEF"},
        "get_collection_images": COLLECTION_FIXTURE,
        "open_darkroom": {"view": "darkroom", "path": "/photos/b.NEF"},
    })
    result = await server._handle_navigate_photo({"direction": "next"})
    text = result[0].text
    assert "Opened image 2 in darkroom" in text
    assert "next photo: 2/3" in text
    assert "b.NEF" in text


@pytest.mark.asyncio
async def test_navigate_photo_previous_opens_neighbor():
    server = DarktableMCPServer()
    server.bridge = _bridge_router({
        "dev_current_image": {"has_image": True, "id": 2, "path": "/photos/b.NEF"},
        "get_collection_images": COLLECTION_FIXTURE,
        "open_darkroom": {"view": "darkroom", "path": "/photos/a.NEF"},
    })
    result = await server._handle_navigate_photo({"direction": "previous"})
    assert "Opened image 1 in darkroom" in result[0].text


@pytest.mark.asyncio
async def test_navigate_photo_stops_at_start():
    server = DarktableMCPServer()
    server.bridge = _bridge_router({
        "dev_current_image": {"has_image": True, "id": 1, "path": "/photos/a.NEF"},
        "get_collection_images": COLLECTION_FIXTURE,
    })
    result = await server._handle_navigate_photo({"direction": "previous"})
    assert "Already at the first photo" in result[0].text


@pytest.mark.asyncio
async def test_navigate_photo_stops_at_end():
    server = DarktableMCPServer()
    server.bridge = _bridge_router({
        "dev_current_image": {"has_image": True, "id": 3, "path": "/photos/c.NEF"},
        "get_collection_images": COLLECTION_FIXTURE,
    })
    result = await server._handle_navigate_photo({"direction": "next"})
    assert "Already at the last photo" in result[0].text


@pytest.mark.asyncio
async def test_navigate_photo_no_image_open():
    server = DarktableMCPServer()
    server.bridge = _bridge_router({"dev_current_image": {"has_image": False}})
    result = await server._handle_navigate_photo({"direction": "next"})
    assert "No image open in darkroom" in result[0].text


@pytest.mark.asyncio
async def test_navigate_photo_current_not_in_collection():
    server = DarktableMCPServer()
    server.bridge = _bridge_router({
        "dev_current_image": {"has_image": True, "id": 999, "path": "/photos/z.NEF"},
        "get_collection_images": COLLECTION_FIXTURE,
    })
    result = await server._handle_navigate_photo({"direction": "next"})
    assert "not in the currently open collection" in result[0].text


@pytest.mark.asyncio
async def test_navigate_photo_rejects_bad_direction():
    server = DarktableMCPServer()
    result = await server._handle_navigate_photo({"direction": "sideways"})
    assert "direction must be" in result[0].text
