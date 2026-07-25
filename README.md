# Darktable MCP Server

A Model Context Protocol (MCP) server that exposes darktable operations
to MCP clients (Claude Desktop, Claude Code, etc.). The AI lives in the
client; this server drives darktable.

## Tools

**Library operations** (require `darktable-mcp install-plugin` and an open darktable session):

- `view_photos(filter?, rating_min?, limit?)` — Browse the library by filename substring and minimum rating. Returns id, filename, **absolute file path**, and rating per match — the path drops straight into `export_images`'s `photo_ids`.
- `get_contact_sheet(offset?, limit?, columns?, filter?, sort?, direction?, thumbnail_width?, background?)` — Render one grid image (JPEG) from the current lighttable collection for batch visual culling, so an agent can page through e.g. 25-photo sheets instead of opening every photo in darkroom. `filter` = `all`/`unrated`/`rated`/`rejected`/`selected`. Read-only — never touches ratings, tags, edit history, or darkroom state. Response includes `next_offset`/`has_more`/`total_matching` for paging, and per-item `position`/`image_id`/`filename` for picking specific photos out of a sheet.
- `rate_photos(photo_ids, rating)` — Apply -1..5 star ratings (-1 = reject, 0 = unrated).
- `import_batch(source_path, recursive?)` — Register a folder as a film roll.
- `list_styles()` — Enumerate installed darktable styles (presets), returning name + description per entry.
- `apply_preset(photo_ids, preset_name)` — Apply a named darktable style to one or more photos. Use `list_styles` first to discover exact names.

**Camera ingest** (headless):

- `import_from_camera(destination?, camera_port?, timeout_seconds?)` — Detect a camera via libgphoto2 and copy photos to a local directory. Auto-merges hybrid setups (one card on PTP, the other mounted as USB Mass-Storage) into a single import — Nikon DSLRs in particular show up that way and the previous behavior silently halved the import.

**Vision-rating workflow** (headless, file-based — no library required, needs `[vision]` extra):

- `extract_previews(source_dir, output_dir?, max_dim?, thumb_dim?, overwrite?)` — Pull auto-rotated JPEG previews + small thumbs out of raws (NEF/CR2/ARW/DNG/...), with an EXIF summary per file. Per-file details (paths, EXIF, errors) land in `<output_dir>/.extract_previews.jsonl`; the tool response keeps only counts and the side-file path so 700+ NEFs don't overflow the agent's context.
- `apply_ratings_batch(source_dir, ratings, log?)` — Write XMP `xmp:Rating` sidecars for a `{stem: rating}` batch + an append-only `ratings.jsonl` log.
- `open_in_darktable(source_dir, rating?, rating_min?, rating_max?)` — Launch the GUI on a folder. Auto-registers as a film roll; pre-applies any rating filter (exact, ≥, ≤, or inner range) via `dt.gui.libs.collect.filter`.

**Export:**

- `export_images(photo_ids, output_path, format, quality?)` — Export to JPEG/PNG/TIFF via `darktable-cli`. Runs in an isolated config dir under `$XDG_CACHE_HOME/darktable-mcp/cli-config/`, so exports work even when the GUI is open (no `database is locked` race against the user's `~/.config/darktable/library.db`). Per-file results land in `<output_path>/.export_images.jsonl`; the tool response is bounded — counts, side-file path, and the first error if any.

**Darkroom editing — scalar params, live preview** (Phase 1, `T1.6`; requires the `darktable-mcp` Lua plugin AND the `darktable.develop.*` C API on branch `agentic-mcp` — see Connect section below for the two ways to get that):

- `open_image_in_darkroom(image_id)` — Enter darkroom on an image (id from `view_photos`). Call this first; the four tools below all act on whichever image is currently open.
- `navigate_photo(direction, sort?, direction_order?)` — Move to the next/previous photo in the currently open collection (same ordering as `get_contact_sheet`) and open it in darkroom, replacing whichever image is open now. Errors clearly at either end instead of wrapping around.
- `list_modules()` — List active processing modules on the open image as `{op, instance, enabled}` (exposure, temperature, colorbalancergb, tonecurve, ...). `op` is what you pass to `get_params`/`set_params`.
- `get_params(op, instance=0)` — Read a module's current scalar/bool/enum fields, each with `{value, min, max, default}`. Ground any proposed edit in the real bounds before calling `set_params`.
- `set_params(op, fields, instance=0)` — Write fields, commit to darkroom history, reprocess. Values are always clamped to `min`/`max` (never rejected); the returned report lists every clamp hit so the AI can say "that's already at max" instead of silently no-opping.
- `get_preview(max_w=1024, max_h=1024)` — PNG of the CURRENT live edit (no DB write). Call after `set_params` to see the result before proposing the next nudge.

Scope: SCALAR fields only (exposure, white balance temperature/tint, contrast, saturation, etc.). Curves/arrays and masks are deferred to a later phase.

## Design rules

Use only the official darktable APIs: `darktable-cli` for export, the Lua API for everything else. No direct `library.db` reads or writes. Tools that return data to the AI must be headless; the GUI may launch only when the tool's purpose is to show the human something.

## Why some tools are parked

`darktable-cli` doesn't load the user's library and `darktable --lua` brings up the full GUI, so there's no headless one-shot path for library reads/writes. Iteration 2 (spec: `docs/superpowers/specs/2026-04-27-ipc-bridge-mvp-design.md`) shipped a long-running Lua plugin loaded into the user's interactive darktable session, with a file-based JSON RPC bridge. The library tools (`view_photos`, `rate_photos`, `import_batch`, `list_styles`, `apply_preset`) all ride on it.

`adjust_exposure` was retired during iteration 3 — see `docs/superpowers/specs/2026-04-28-iter3-design.md`. At the time, the stock darktable Lua API exposed neither `image.modules` nor `image.history`, and `dt.gui.action` required an active darkroom view (single-image, GUI-driven). That gap is what Phase 1 (`../PLAN.md` §3–§5, branch `agentic-mcp`) closed by adding a real `darktable.develop.*` Lua namespace in C (`get_params`/`set_params` with introspection-backed min/max clamping, plus a live no-DB `preview()`) — the darkroom-editing tools above ride on it. `dt.gui.action` remains parked as a de-risking spike (T0.3), not the production path.

## Installation

```bash
pip install darktable-mcp
# Optional: vision-rating workflow extras
pip install 'darktable-mcp[vision]'
# Install the Lua plugin into ~/.config/darktable/, then restart darktable
darktable-mcp install-plugin
```

You also need `darktable` (with `darktable-cli`) on `PATH`. The `[vision]` extra pulls in `rawpy`, `Pillow`, and `pyexiv2`, which need system `libraw` and `libexiv2`.

## Configuration

Add to your Claude Desktop config:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "darktable": {
      "command": "darktable-mcp"
    }
  }
}
```

## Connect from Claude Desktop / Claude Code

The MCP server is just a stdio process — point your client at it and make sure a darktable
with the bridge loaded is running before you use it.

**1. The server command.** Same for Claude Desktop or Claude Code:

- Installed via `pip install darktable-mcp`: command is `darktable-mcp` (see Configuration above
  for the Desktop `mcpServers` JSON; for Claude Code, `claude mcp add darktable -- darktable-mcp`).
- Running from this checkout without installing: `python -m darktable_mcp` from `darktable-mcp/`
  (needs `mcp` importable — see Installation; this repo's `mcp` dependency requires **Python
  3.10+**, so on a distro whose system `python3` is older, create a venv with a newer interpreter,
  e.g. `uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -e .`, and point the
  client's `command` at `.venv/bin/python` with `args: ["-m", "darktable_mcp"]`).

**2. darktable must be running with the bridge loaded.** The server talks to darktable only
through the file-based JSON bridge (`~/.cache/darktable-mcp/request-*.json` /
`response-*.json` by default, or `$XDG_CACHE_HOME/darktable-mcp` if set) — every tool call times
out with "darktable not running, or plugin not loaded" until this is true.

- **On your real workstation** (the case the user follows day to day): install the patched
  darktable build (the one with the `darktable.develop.*` C API from `agentic-mcp`, not stock
  darktable), then either run `darktable-mcp install-plugin` (copies
  `darktable_mcp/lua/darktable_mcp.lua` into `~/.config/darktable/lua/` and adds
  `require "darktable_mcp"` to `~/.config/darktable/luarc`) or add that `require` line by hand if
  you manage `luarc` yourself. Start darktable normally; the darkroom-editing tools only do
  anything once an image is actually open in darkroom (`open_image_in_darkroom` gets you there).
- **Containerized / dev harness** (what this repo's own acceptance tests use, no real display or
  library needed): `docker/build.sh` once to build the image, then
  `docker/run-dt-bridge.sh start [run_dir]` launches a detached Xvfb darktable with the bridge
  wired in and prints `RUN_DIR`; `docker/run-dt-bridge.sh stop <run_dir>` tears it down. Because
  darktable runs inside the container, the bridge's cache dir and `get_preview()`'s returned PNG
  path are container paths under `/run`; point the server at the run by setting
  `XDG_CACHE_HOME=<run_dir>/cache-mcp` (so it reads/writes the same bind-mounted request/response
  files) and `DARKTABLE_MCP_RUN_DIR=<run_dir>` (so `get_preview()` remaps `/run/...` back to a
  path on the host you can actually open).

## Vision-rating workflow

When darktable's library doesn't yet know about your shoot — typically straight off a card — you can rate by vision before any import:

1. `extract_previews` writes auto-rotated JPEGs and an EXIF summary so the client can iterate efficiently.
2. The client reads previews, decides ratings, and calls `apply_ratings_batch` to write XMP sidecars next to the raws.
3. `open_in_darktable` launches the GUI with the folder as a film roll, lighttable filtered to the rating range you want.

No SQLite poking, no half-imported state, no GUI launch until step 3.

## Requirements

- Python 3.10+ (the `mcp` package pulled in by `darktable_mcp/__init__.py` requires it; on a
  system whose default `python3` is older, use a venv with a newer interpreter — see Connect
  section)
- darktable 4.0+ (with `darktable-cli` on `PATH`); the darkroom-editing tools need the
  `agentic-mcp` branch build (adds the `darktable.develop.*` C API) — plain upstream darktable
  only serves the library/camera/export tools
- An MCP-compatible client (Claude Desktop, Claude Code, etc.)

## Contributing

Contributions welcome. Any change that reads or writes `library.db` directly will be rejected.

## License

MIT — see `LICENSE`.
