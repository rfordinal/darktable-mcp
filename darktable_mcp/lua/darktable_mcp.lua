-- darktable_mcp: long-running plugin that exposes view_photos, rate_photos,
-- tag_photo, list_collections, list_photos_in_collection (tag-backed --
-- darktable's Lua API has no separate "collection" object), and the
-- darktable.develop.* darkroom-editing bridge (open_darkroom,
-- dev_version, dev_active_modules, dev_current_image, dev_get_conf_string, dev_get_params,
-- dev_set_params, dev_preview, dev_history_count, dev_enable_module,
-- dev_add_instance, dev_get_viewport, dev_add_path_mask,
-- dev_remove_last_instance, dev_set_raster_source) to the Python MCP server via file-based JSON
-- requests.
--
-- Loaded via `require "darktable_mcp"` from ~/.config/darktable/luarc.
-- Spawns a worker via dt.control.dispatch that polls
-- ~/.cache/darktable-mcp/ every ~100ms for request-*.json files,
-- dispatches them to the method registry, and writes
-- response-<uuid>.json. See:
--   docs/superpowers/specs/2026-04-27-ipc-bridge-mvp-design.md
--   ../../PLAN.md §3 (darktable.develop.* Lua API) and §5 T1.6

local dt = require("darktable")

-- ---- JSON encode/decode (minimal, MVP-only) --------------------------------
-- darktable Lua does not bundle a JSON library reliably. Inline a tiny
-- encoder/decoder sufficient for our request/response shapes.

local json = {}

local function encode_value(v)
  local t = type(v)
  if t == "nil" then return "null"
  elseif t == "boolean" then return v and "true" or "false"
  elseif t == "number" then return tostring(v)
  elseif t == "string" then
    local escaped = v:gsub('\\', '\\\\'):gsub('"', '\\"'):gsub('\n', '\\n'):gsub('\r', '\\r'):gsub('\t', '\\t')
                     :gsub('\b', '\\b'):gsub('\f', '\\f')
    -- Escape any remaining control bytes (0x00-0x1F) as \u00XX.
    escaped = escaped:gsub('[%z\1-\31]', function(c)
      return string.format('\\u%04x', string.byte(c))
    end)
    return '"' .. escaped .. '"'
  elseif t == "table" then
    -- Detect array vs object by checking for sequential integer keys.
    local n, max = 0, 0
    for k in pairs(v) do
      n = n + 1
      if type(k) == "number" and k > max then max = k end
    end
    if n == max and n > 0 then
      local parts = {}
      for i = 1, n do parts[i] = encode_value(v[i]) end
      return "[" .. table.concat(parts, ",") .. "]"
    elseif n == 0 then
      return "[]"
    else
      local parts = {}
      for k, val in pairs(v) do
        table.insert(parts, encode_value(tostring(k)) .. ":" .. encode_value(val))
      end
      return "{" .. table.concat(parts, ",") .. "}"
    end
  end
  error("cannot encode value of type " .. t)
end

function json.encode(v) return encode_value(v) end

-- Minimal recursive-descent decoder. Adequate for plain JSON requests.
local function skip_ws(s, i)
  while i <= #s and (s:sub(i,i) == " " or s:sub(i,i) == "\t" or s:sub(i,i) == "\n" or s:sub(i,i) == "\r") do
    i = i + 1
  end
  return i
end

local decode_value
local function decode_string(s, i)
  assert(s:sub(i,i) == '"', "expected string at " .. i)
  i = i + 1
  local out = {}
  while i <= #s do
    local c = s:sub(i,i)
    if c == '"' then return table.concat(out), i + 1
    elseif c == "\\" then
      local esc = s:sub(i+1, i+1)
      if esc == "n" then table.insert(out, "\n"); i = i + 2
      elseif esc == "r" then table.insert(out, "\r"); i = i + 2
      elseif esc == "t" then table.insert(out, "\t"); i = i + 2
      elseif esc == "b" then table.insert(out, "\b"); i = i + 2
      elseif esc == "f" then table.insert(out, "\f"); i = i + 2
      elseif esc == '"' or esc == "\\" or esc == "/" then table.insert(out, esc); i = i + 2
      elseif esc == "u" then
        -- \uXXXX: parse 4 hex digits, emit UTF-8 bytes.
        local hex = s:sub(i+2, i+5)
        if #hex ~= 4 or not hex:match("^%x%x%x%x$") then
          error("malformed \\u escape at " .. i)
        end
        local cp = tonumber(hex, 16)
        -- Encode codepoint as UTF-8.
        if cp < 0x80 then
          table.insert(out, string.char(cp))
        elseif cp < 0x800 then
          table.insert(out, string.char(
            0xC0 + math.floor(cp / 0x40),
            0x80 + (cp % 0x40)
          ))
        else
          table.insert(out, string.char(
            0xE0 + math.floor(cp / 0x1000),
            0x80 + (math.floor(cp / 0x40) % 0x40),
            0x80 + (cp % 0x40)
          ))
        end
        i = i + 6
      else error("unsupported escape \\" .. esc)
      end
    else
      table.insert(out, c)
      i = i + 1
    end
  end
  error("unterminated string")
end

local function decode_number(s, i)
  local start = i
  if s:sub(i,i) == "-" then i = i + 1 end
  while i <= #s and s:sub(i,i):match("[%d%.eE%-%+]") do i = i + 1 end
  return tonumber(s:sub(start, i-1)), i
end

local function decode_array(s, i)
  assert(s:sub(i,i) == "[")
  i = i + 1
  i = skip_ws(s, i)
  local out = {}
  if s:sub(i,i) == "]" then return out, i + 1 end
  while true do
    local v
    v, i = decode_value(s, i)
    table.insert(out, v)
    i = skip_ws(s, i)
    local c = s:sub(i,i)
    if c == "," then i = i + 1; i = skip_ws(s, i)
    elseif c == "]" then return out, i + 1
    else error("expected , or ] at " .. i)
    end
  end
end

local function decode_object(s, i)
  assert(s:sub(i,i) == "{")
  i = i + 1
  i = skip_ws(s, i)
  local out = {}
  if s:sub(i,i) == "}" then return out, i + 1 end
  while true do
    local k
    k, i = decode_string(s, i)
    i = skip_ws(s, i)
    assert(s:sub(i,i) == ":", "expected : at " .. i)
    i = skip_ws(s, i + 1)
    local v
    v, i = decode_value(s, i)
    out[k] = v
    i = skip_ws(s, i)
    local c = s:sub(i,i)
    if c == "," then i = i + 1; i = skip_ws(s, i)
    elseif c == "}" then return out, i + 1
    else error("expected , or } at " .. i)
    end
  end
end

decode_value = function(s, i)
  i = skip_ws(s, i)
  local c = s:sub(i,i)
  if c == "{" then return decode_object(s, i)
  elseif c == "[" then return decode_array(s, i)
  elseif c == '"' then return decode_string(s, i)
  elseif c == "t" and s:sub(i, i+3) == "true" then return true, i + 4
  elseif c == "f" and s:sub(i, i+4) == "false" then return false, i + 5
  elseif c == "n" and s:sub(i, i+3) == "null" then return nil, i + 4
  else return decode_number(s, i)
  end
end

function json.decode(s)
  local v = decode_value(s, 1)
  return v
end

-- ---- Method registry -------------------------------------------------------

local methods = {}

-- darktable's image.path is the parent directory; image.filename is the
-- bare basename. Callers want a single absolute file path they can hand
-- straight to export_images / Read / etc., so join them once here.
local function _image_full_path(image)
  local dir = image.path or ""
  if #dir > 0 and dir:sub(-1) ~= "/" then dir = dir .. "/" end
  return dir .. (image.filename or "")
end

-- dt.collection is the currently active lighttable view (whatever filter/
-- filmroll/collect-module rule is open right now); dt.database is the WHOLE
-- library regardless of what's open. Default to the former so "browse my
-- photos" means "what I'm actually looking at", not a random slice of
-- everything ever imported. scope="library" opts back into the old
-- whole-library behavior for callers that explicitly want it.
methods.view_photos = function(p)
  p = p or {}
  local out, count = {}, 0
  local limit = p.limit or 100
  local filter = p.filter or ""
  local rating_min = p.rating_min
  local source = (p.scope == "library") and dt.database or dt.collection
  for _, image in ipairs(source) do
    if count >= limit then break end
    local include = true
    if rating_min and (image.rating or 0) < rating_min then include = false end
    if include and filter ~= "" then
      local ok = string.find(string.lower(image.filename), string.lower(filter), 1, true)
      if not ok then include = false end
    end
    if include then
      table.insert(out, {
        id = tostring(image.id),
        filename = image.filename,
        path = _image_full_path(image),
        sidecar = image.sidecar,
        rating = image.rating or 0,
      })
      count = count + 1
    end
  end
  return out
end

-- Full, unfiltered dump of the current collection for get_contact_sheet
-- (T?.?, contact-sheet spec): filter/sort/offset/limit all happen in Python
-- on this raw list -- view_photos already does filter+limit lua-side, but
-- contact_sheet needs total_matching (post-filter, pre-page count) and
-- offset-based pagination, which only work if the whole collection crosses
-- the bridge once rather than being re-fetched per page. dt.gui.selection()
-- is otherwise only used internally by open_darkroom (see below); here it's
-- surfaced per-image so Python's filter="selected" has something to check.
methods.get_collection_images = function(p)
  p = p or {}
  local source = (p.scope == "library") and dt.database or dt.collection
  local selected_ids = {}
  for _, img in ipairs(dt.gui.selection()) do
    selected_ids[tostring(img.id)] = true
  end
  local out = {}
  for _, image in ipairs(source) do
    table.insert(out, {
      id = tostring(image.id),
      filename = image.filename,
      path = _image_full_path(image),
      rating = image.rating or 0,
      capture_time = image.exif_datetime_taken or "",
      selected = selected_ids[tostring(image.id)] or false,
    })
  end
  return out
end

methods.rate_photos = function(p)
  p = p or {}
  local updated = 0
  for _, photo_id in ipairs(p.photo_ids or {}) do
    local image = dt.database.get_image(tonumber(photo_id))
    if image then
      image.rating = p.rating
      updated = updated + 1
    end
  end
  return {updated = updated}
end

-- Tags are darktable's only persistent, user-named grouping of photos
-- (there is no separate "collection" object in the Lua API) so
-- list_collections / list_photos_in_collection surface dt.tags as the
-- practical equivalent of Lightroom-style collections.
methods.tag_photo = function(p)
  p = p or {}
  local photo_ids = p.photo_ids or {}
  if #photo_ids == 0 then error("tag_photo: photo_ids required and non-empty") end
  local add_names = p.tags or {}
  local remove_names = p.remove_tags or {}
  if #add_names == 0 and #remove_names == 0 then
    error("tag_photo: at least one of tags or remove_tags required")
  end

  local created = {}
  local add_tags = {}
  for _, name in ipairs(add_names) do
    local tag = dt.tags.find(name)
    if not tag then
      tag = dt.tags.create(name)
      table.insert(created, name)
    end
    table.insert(add_tags, tag)
  end

  local remove_tags = {}
  for _, name in ipairs(remove_names) do
    local tag = dt.tags.find(name)
    if tag then table.insert(remove_tags, tag) end
  end

  local updated, missing_photos = 0, {}
  for _, photo_id in ipairs(photo_ids) do
    local image = dt.database.get_image(tonumber(photo_id))
    if image then
      for _, tag in ipairs(add_tags) do image:attach_tag(tag) end
      for _, tag in ipairs(remove_tags) do image:detach_tag(tag) end
      updated = updated + 1
    else
      table.insert(missing_photos, tostring(photo_id))
    end
  end

  return {updated = updated, missing_photos = missing_photos, tags_created = created}
end

-- AI assessments/notes are stored in the standard dc:description xmp field
-- (image.description in darktable's Lua API), not a custom field, so they
-- round-trip through any xmp-aware tool and show up in darktable's own
-- metadata panel like a human-written caption would.
methods.set_photo_note = function(p)
  p = p or {}
  if not p.photo_id then error("set_photo_note: photo_id required") end
  local image = dt.database.get_image(tonumber(p.photo_id))
  if not image then return {updated = 0} end
  image.description = p.note or ""
  return {updated = 1}
end

methods.get_photo_note = function(p)
  p = p or {}
  if not p.photo_id then error("get_photo_note: photo_id required") end
  local image = dt.database.get_image(tonumber(p.photo_id))
  if not image then return {found = false} end
  return {found = true, note = image.description or ""}
end

methods.list_collections = function(p)
  p = p or {}
  local filter = p.filter or ""
  local out = {}
  for _, tag in ipairs(dt.tags) do
    local include = true
    if filter ~= "" then
      local ok = string.find(string.lower(tag.name), string.lower(filter), 1, true)
      if not ok then include = false end
    end
    if include then
      table.insert(out, {name = tag.name, count = #tag})
    end
  end
  return {collections = out, count = #out}
end

methods.list_photos_in_collection = function(p)
  p = p or {}
  local name = p.collection
  if not name or name == "" then error("list_photos_in_collection: collection required") end
  local tag = dt.tags.find(name)
  if not tag then return {found = false, collection = name, photos = {}, count = 0} end

  local limit = p.limit or 1000
  local out = {}
  for _, image in ipairs(tag:get_tagged_images()) do
    if #out >= limit then break end
    table.insert(out, {
      id = tostring(image.id),
      filename = image.filename,
      path = _image_full_path(image),
      rating = image.rating or 0,
    })
  end
  return {found = true, collection = name, photos = out, count = #out}
end

methods.import_batch = function(p)
  p = p or {}
  local source_path = p.source_path
  if not source_path or source_path == "" then
    error("import_batch: source_path required")
  end
  local recursive = p.recursive
  if recursive == nil then recursive = true end
  -- dt.database.import takes only a path string in darktable 5.x;
  -- recursion follows the "recurse_directories" preference. The return
  -- value is heterogeneous: a dt_lua_image_t for a single file, a
  -- dt_lua_film_t (the registered film roll) for a directory. Darktable
  -- scans the folder asynchronously, so the image count is not yet
  -- available when this call returns.
  local imported = dt.database.import(source_path)

  -- For a stub-table return (used in unit tests) `#imported` is the
  -- definitive count.
  local count
  if type(imported) == "table" then
    count = #imported
  else
    -- Real darktable: poll dt.database for images whose film.path equals
    -- source_path. The scan runs on a background thread; sleep+retry up
    -- to ~3s before giving up.
    count = 0
    local attempts = 0
    while attempts < 30 do
      count = 0
      for _, image in ipairs(dt.database) do
        local film = image.film
        if film and film.path == source_path then
          count = count + 1
        end
      end
      if count > 0 then break end
      if dt.control and dt.control.sleep then
        dt.control.sleep(100)
      end
      attempts = attempts + 1
    end
    -- Final fallback: if the scan turned up nothing but the return value
    -- is non-nil, treat as 1 (single-file import succeeded).
    if count == 0 and imported ~= nil then count = 1 end
  end

  return {imported = count, source_path = source_path, recursive = recursive}
end

methods.list_styles = function(p)
  local out = {}
  for _, style in ipairs(dt.styles) do
    table.insert(out, {
      name = style.name,
      description = style.description,
    })
  end
  return {styles = out, count = #out}
end

methods.apply_preset = function(p)
  p = p or {}
  local preset_name = p.preset_name
  local photo_ids = p.photo_ids or {}
  if not preset_name or preset_name == "" then
    error("apply_preset: preset_name required")
  end
  if #photo_ids == 0 then
    error("apply_preset: photo_ids required and non-empty")
  end

  -- Linear scan to resolve preset_name (string lookup unavailable on dt.styles).
  local style = nil
  for _, s in ipairs(dt.styles) do
    if s.name == preset_name then style = s; break end
  end
  if not style then
    error(string.format("apply_preset: style %q not found (use list_styles)", preset_name))
  end

  local applied = 0
  local missed = {}
  for _, photo_id in ipairs(photo_ids) do
    local image = dt.database.get_image(tonumber(photo_id))
    if image then
      image:apply_style(style)
      applied = applied + 1
    else
      table.insert(missed, tostring(photo_id))
    end
  end
  return {applied = applied, missed = missed, preset_name = preset_name}
end

-- ---- darktable.develop.* bridge (T1.6) -------------------------------------
-- Promoted from darktable-mcp/spike/spike_methods.lua (T0.3/T1.1-T1.5 probes)
-- into the clean bridge once the underlying C API stabilized. These wrap the
-- NEW Lua namespace `darktable.develop` (src/lua/develop.c, registered in
-- src/lua/init.c) added on branch agentic-mcp: version/active_modules/
-- get_params/set_params/preview/history_count, all scalar-only for now (see
-- PLAN.md §3.1 Risk #2 -- arrays/curves and masks are out of scope here).
--
-- open_darkroom is pure-Lua (no C change): it drives the selection +
-- current_view + async-poll technique sketched by the T0.3 spike, since
-- headless darkroom entry has no mouse to resolve dt_act_on_get_main_image()
-- via hover (see common/act_on.c:_get_main_image_hover). CORRECTION
-- (2026-07-24 bugreport): this comment used to claim the spike "proved" this
-- works -- it didn't. spike/run_spike.py's own sub-goal A branch explicitly
-- anticipates open_darkroom possibly NOT switching view and treats that as a
-- valid (documented-negative) spike outcome; no pass was ever recorded. This
-- worker_loop also runs on darktable's Lua thread pool (see call.c
-- stacked_job_queue / GThreadPool), not the GTK main thread, so
-- dt.gui.current_view(view) only *schedules* the switch on the GTK context
-- (g_main_context_invoke(), control/control.c:630) rather than performing it
-- inline -- hence the poll below. But scheduling the view switch is not
-- sufficient: views/darkroom.c's try_enter() independently re-resolves the
-- target image via dt_act_on_get_main_image() (common/act_on.c:650), which
-- can silently prefer mouseover/stale active_images over the Lua selection
-- we just set, or require the image to be part of the CURRENT lighttable
-- collection filter -- see the pre-check below, added after a real-world
-- failure where the switch never happened and try_enter() gave no error
-- back to Lua at all (darkroom.c:1150, "no image to open!" is GUI-log only).

methods.open_darkroom = function(p)
  p = p or {}
  local image_id = tonumber(p.image_id)
  if not image_id then error("open_darkroom: image_id required") end
  local image = dt.database.get_image(image_id)
  if not image then error("open_darkroom: image not found: " .. tostring(image_id)) end

  dt.gui.selection({ image })

  -- Diagnostic pre-check (bugreport 2026-07-24): try_enter() in
  -- views/darkroom.c does NOT read this Lua selection directly -- it
  -- resolves the target image via dt_act_on_get_main_image()
  -- (common/act_on.c:650), which can silently prefer mouseover or stale
  -- view_manager->active_images over the selection we just set (if the
  -- "activate images by" preference, plugins/lighttable/act_on, is hover
  -- mode -- act_on.c:34), and either way requires the image to be part of
  -- the CURRENT lighttable collection filter (its selection-table fallback
  -- JOINs against memory.collected_images -- act_on.c:582,632). If any of
  -- that fails, try_enter() just logs "no image to open!" and aborts with
  -- no error back to Lua (darkroom.c:1150), which used to show up here as a
  -- silent 3000ms timeout stuck in "lighttable" with zero diagnostics.
  -- dt.gui.action_images (lua/gui.c:85) calls the exact same
  -- dt_act_on_get_images() resolution try_enter() uses, so checking it here
  -- catches all three failure modes up front instead of burning the full
  -- poll timeout on a switch that could never succeed.
  local action_ids = {}
  for _, img in ipairs(dt.gui.action_images) do
    table.insert(action_ids, img.id)
  end
  local resolves_to_target = (#action_ids == 1 and action_ids[1] == image.id)
  if not resolves_to_target then
    return {
      view = dt.gui.current_view().name,
      requested_image_id = image_id,
      selection_count = #dt.gui.selection(),
      waited_ms_for_view_switch = 0,
      action_images = action_ids,
      diagnostic = "selection did not resolve to the requested image via darktable's " ..
        "act-on resolution (dt.gui.action_images = {" .. table.concat(action_ids, ",") ..
        "}); darkroom entry would silently fail without ever switching view. Likely " ..
        "cause: the 'activate images by' preference is set to mouseover and something " ..
        "else is hovered/active, OR image " .. tostring(image_id) .. " is not part of " ..
        "the current lighttable collection filter.",
    }
  end

  -- Bugreport 2026-07-25: switching current_view to darkroom while ALREADY
  -- IN darkroom (opening a second image right after the first) does NOT
  -- reload the new image. views/view.c's dt_view_manager_switch_by_view
  -- only calls the target view's enter() -- which is what calls
  -- dt_dev_load_image() for the new image_storage.id, views/darkroom.c:3860
  -- -- when new_view != old_view (view.c:443-444: "if(new_view != old_view
  -- && new_view->enter) new_view->enter(new_view);"). try_enter() alone
  -- (which DOES re-run and sets image_storage.id) is not enough: the actual
  -- pixelpipe/history/GUI reload lives in enter(), so without a real view
  -- transition darkroom keeps rendering the previous image while this
  -- method still reports "view=darkroom" as if it had switched. Force a
  -- genuine transition by bouncing through lighttable first so enter() is
  -- guaranteed to run again for the new image.
  local bounced = false
  if dt.gui.current_view().name == "darkroom" then
    bounced = true
    dt.gui.current_view(dt.gui.views.lighttable)
    local left = dt.gui.current_view()
    local left_wait = 0
    while left.name == "darkroom" and left_wait < 3000 do
      if dt.control and dt.control.sleep then
        dt.control.sleep(100)
        left_wait = left_wait + 100
      else
        break
      end
      left = dt.gui.current_view()
    end
  end

  dt.gui.current_view(dt.gui.views.darkroom)

  local current = dt.gui.current_view()
  local waited_ms = 0
  while current.name ~= "darkroom" and waited_ms < 3000 do
    if dt.control and dt.control.sleep then
      dt.control.sleep(100)
      waited_ms = waited_ms + 100
    else
      break
    end
    current = dt.gui.current_view()
  end

  return {
    view = current.name,
    requested_image_id = image_id,
    selection_count = #dt.gui.selection(),
    waited_ms_for_view_switch = waited_ms,
    bounced_through_lighttable = bounced,
    -- Absolute file path of the opened image (dir + filename joined via the
    -- module-local _image_full_path helper, same one view_photos uses).
    -- T3.3's mask_raster needs this to feed the actual image file to the
    -- matting sidecar without a dedicated new bridge method.
    path = _image_full_path(image),
  }
end

-- Version handshake (PLAN.md §3.5): {api, dt_lua_api, min_bridge}. Callers
-- can refuse to run if the C binding is older than the bridge expects.
methods.dev_version = function(p)
  return dt.develop.version()
end

-- {op, instance, id, multi_name, enabled, has_introspection} per active
-- module on the image currently open in darkroom.
methods.dev_active_modules = function(p)
  local mods = dt.develop.active_modules()
  local out = {}
  for i, m in ipairs(mods) do
    out[i] = {
      op = m.op,
      instance = m.instance,
      id = m.id,
      multi_name = m.multi_name,
      enabled = m.enabled,
      has_introspection = m.has_introspection,
    }
  end
  -- count disambiguates the empty case (Lua {} would JSON-encode ambiguously)
  return { count = #out, modules = out }
end

-- Report the image currently open in darkroom: {has_image, id, path, filename}.
-- Lets the server resolve the source file for an image the user opened BY HAND
-- in the GUI (no open_darkroom call, so no MCP-side path cache). Returns the C
-- result verbatim; has_image=false when no darkroom image is loaded. Also adds
-- `sidecar` (2026-07-31 bugreport: export_images silently exported the WRONG
-- duplicate/version because it never told darktable-cli which .xmp to use,
-- so darktable-cli fell back to its own auto-detect of the base/version-0
-- sidecar) -- `image.sidecar` is a stock darktable Lua field (already
-- version/duplicate-aware, src/lua/image.c) the C current_image() binding
-- itself doesn't return; fetched here in pure Lua via dt.database.get_image,
-- no C change needed.
methods.dev_current_image = function(p)
  local result = dt.develop.current_image()
  if result.has_image then
    local image = dt.database.get_image(result.id)
    if image then result.sidecar = image.sidecar end
  end
  return result
end

-- Read-only passthrough to an arbitrary core dt_conf key, e.g.
-- "plugins/darkroom/lut3d/def_path" (the lut3d module's configured LUT root
-- directory -- same key the UI file-chooser widget reads). Returns the raw
-- string verbatim ("" if the key resolves to an empty default -- never nil).
methods.dev_get_conf_string = function(p)
  p = p or {}
  local key = p.key
  if not key or key == "" then error("dev_get_conf_string: key required") end
  return dt.develop.get_conf_string(key)
end

-- Typed field map for one module instance (PLAN.md §3.1): scalar fields come
-- back as {value, min, max, default}. Returns the C result verbatim --
-- {op, instance, id, fields=...} on success, {error=...} if op/instance is
-- unknown.
methods.dev_get_params = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_get_params: op required") end
  local instance = tonumber(p.instance) or 0
  return dt.develop.get_params(op, instance)
end

-- Write fields, push history, reprocess. `fields` is a plain Lua table of
-- field=value. Returns the clamp report verbatim (PLAN.md §3.2):
-- {ok, applied, clamped, unknown_fields} -- out-of-range values are clamped
-- to introspection Min/Max (never rejected) and every clamp is reported.
methods.dev_set_params = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_set_params: op required") end
  local instance = tonumber(p.instance) or 0
  local fields = p.fields
  if type(fields) ~= "table" then error("dev_set_params: fields table required") end
  return dt.develop.set_params(op, instance, fields)
end

-- Read-only blend_params snapshot (opacity, mask_mode, blend_mode). Separate
-- from dev_get_params: blend_params is a plain struct outside module
-- introspection (LUT tooling, 2026-07-26 -- see get_blend_params_cb).
methods.dev_get_blend_params = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_get_blend_params: op required") end
  local instance = tonumber(p.instance) or 0
  return dt.develop.get_blend_params(op, instance)
end

-- Write blend_params fields (opacity 0..100, blend_mode raw int,
-- enable_uniform_blend convenience bool) and commit to history. See
-- set_blend_params_cb for the enable_uniform_blend all-or-nothing contract.
methods.dev_set_blend_params = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_set_blend_params: op required") end
  local instance = tonumber(p.instance) or 0
  local fields = p.fields
  if type(fields) ~= "table" then error("dev_set_blend_params: fields table required") end
  return dt.develop.set_blend_params(op, instance, fields)
end

-- Read-only history entry count for the open image (used to confirm history
-- coalescing per §3.3 and that preview() never writes to it).
methods.dev_history_count = function(p)
  return dt.develop.history_count()
end

-- PNG of the CURRENT LIVE darkroom edit, grabbed from preview_pipe->backbuf,
-- NO DB write (PLAN.md §3.4). Returns the C result verbatim:
-- {status="ok", path, width, height} | {status="processing", stale_preview,
-- ...} | {error=...}. `path` is a container path under $XDG_CACHE_HOME
-- (/run/cache-mcp in the docker bridge); the Python server remaps
-- /run -> the bridge run_dir so the host-side MCP client can read the file.
--
-- Optional `region` = {x,y,w,h} normalized 0..1 of the visible frame, passed
-- through to dt.develop.preview(max_w, max_h, x, y, w, h) for a full-detail
-- crop render (grain/sharpen/noise inspection). Omit for the full frame.
--
-- Optional `viewport` (2026-07-31 fix, set-viewport-design follow-up): "main"
-- | "preview2" reads dev->full.pipe / dev->preview2.pipe's OWN backbuf
-- instead of the default dev->preview_pipe -- see dt.develop.preview's own
-- long doc comment in src/lua/develop.c for why this is the fix that makes
-- set_viewport's zoom actually show up in captured pixels (preview_pipe has
-- a fixed native resolution entirely decoupled from any darkroom zoom).
-- Omitted: zero change from the original preview_pipe behavior.
methods.dev_preview = function(p)
  p = p or {}
  local max_w = tonumber(p.max_w) or 0
  local max_h = tonumber(p.max_h) or 0
  local region = p.region
  local rx, ry, rw, rh
  if type(region) == "table" and region.x ~= nil and region.y ~= nil
     and region.w ~= nil and region.h ~= nil then
    rx = tonumber(region.x)
    ry = tonumber(region.y)
    rw = tonumber(region.w)
    rh = tonumber(region.h)
  end
  return dt.develop.preview(max_w, max_h, rx, ry, rw, rh, p.viewport)
end

-- Toggle a module instance on/off and commit to history (PLAN.md T1.7).
-- Many modules ship OFF by default (grain, sharpen, vignette, tonecurve,
-- ...) and produce no visible effect until enabled -- this is the missing
-- prerequisite step before set_params on those. Returns the C result
-- verbatim: {ok, op, instance, enabled} | {error=...}.
methods.dev_enable_module = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_enable_module: op required") end
  local instance = tonumber(p.instance) or 0
  local enabled = p.enabled
  if enabled == nil then enabled = true end
  return dt.develop.enable_module(op, instance, enabled and true or false)
end

-- Add a new masked/parametric instance of a module (mirrors the GUI
-- "new instance" action) -- the base step for local edits (dodge/burn on a
-- second exposure instance, a second sharpen for a specific area, etc).
-- Returns the C result verbatim: {ok, op, instance=<new multi_priority>,
-- base_instance, multi_name} | {error=...}.
-- `fields` (optional): initial param overrides applied to the new instance
-- right after duplication, via the same introspection write path as
-- dev_set_params -- see add_instance_cb's doc comment (src/lua/develop.c) for
-- why this collapses into ONE history entry instead of the duplicate's own
-- commit plus a separate later dev_set_params call. Use this for modules
-- whose reload_defaults() gives a fresh instance different defaults than the
-- base instance it was cloned from (exposure's compensate_exposure_bias/
-- compensate_hilite_pres is the motivating case) -- dt_iop_gui_duplicate's
-- copy_params=TRUE clobbers those with the BASE instance's values otherwise.
methods.dev_add_instance = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_add_instance: op required") end
  local fields = p.fields
  if fields ~= nil and type(fields) ~= "table" then
    error("dev_add_instance: fields must be a table if given")
  end
  return dt.develop.add_instance(op, fields)
end

-- Build a drawn PATH mask from a normalized polygon and attach it to a
-- module's blend so the module's effect is restricted to that region
-- (PLAN.md T2.2, promoted from darktable-mcp/spike/spike_methods.lua once
-- the underlying C binding (add_path_mask_cb, src/lua/develop.c) stabilized).
-- `points` is a Lua array of {x=..,y=..} normalized 0..1 (>=3 nodes,
-- typically the polygon returned by the segmentation sidecar via T2.3's
-- mask_object). Returns the C result verbatim: {ok, op, instance, formid,
-- mask_id, points=<node count>, opacity, feather, smooth, mask_mode} |
-- {error=...}. `feather` (optional) is a fraction of the mask bbox; `smooth`
-- (optional bool, default true) uses Catmull-Rom bezier handles so curved
-- subjects are not faceted. nil args fall back to the C-side defaults.
methods.dev_add_path_mask = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_add_path_mask: op required") end
  local instance = tonumber(p.instance) or 0
  local points = p.points
  if type(points) ~= "table" then error("dev_add_path_mask: points array required") end
  local opacity = tonumber(p.opacity)
  if opacity == nil then opacity = 1.0 end
  local feather = tonumber(p.feather) -- nil -> C default (0.02)
  local smooth
  if p.smooth ~= nil then smooth = p.smooth and true or false end -- nil -> C default (true)
  return dt.develop.add_path_mask(op, instance, points, opacity, feather, smooth)
end

-- Retouch module: local heal/clone shapes tied to the module's own wavelet
-- scale + rt_forms array (dt.develop.retouch_add_shape/delete_shape/
-- list_shapes, src/lua/develop.c) -- distinct from dev_add_path_mask's
-- generic "restrict this module's blend to a region". Only circle shapes /
-- heal+clone algorithms are supported this phase; ellipse/path/brush and
-- blur/fill are a later phase (see PLAN.md).
methods.dev_retouch_add_shape = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_retouch_add_shape: op required") end
  local instance = tonumber(p.instance) or 0
  if p.shape_type and p.shape_type ~= "circle" then
    error("dev_retouch_add_shape: only shape_type 'circle' supported in this build")
  end
  local algorithm = p.algorithm
  if not algorithm or algorithm == "" then error("dev_retouch_add_shape: algorithm required") end
  local target = p.target
  if type(target) ~= "table" or tonumber(target.x) == nil or tonumber(target.y) == nil then
    error("dev_retouch_add_shape: target {x,y} required")
  end
  local source = p.source
  if type(source) ~= "table" or tonumber(source.x) == nil or tonumber(source.y) == nil then
    error("dev_retouch_add_shape: source {x,y} required for heal/clone")
  end
  local radius = tonumber(p.radius)
  if radius == nil then error("dev_retouch_add_shape: radius required") end
  local feather = tonumber(p.feather) or 0.0
  local scale = tonumber(p.wavelet_scale) -- nil -> C default (module's curr_scale)
  local opacity = tonumber(p.opacity)
  if opacity == nil then opacity = 1.0 end
  return dt.develop.retouch_add_shape(op, instance, algorithm,
    tonumber(target.x), tonumber(target.y), radius, feather,
    tonumber(source.x), tonumber(source.y), scale, opacity)
end

-- Move/resize an existing shape in place (same formid) -- see
-- dt.develop.retouch_update_shape's doc comment in src/lua/develop.c for why
-- this is a masks-history commit, distinct from retouch_add_shape's
-- iop-params-history one. algorithm/wavelet_scale/opacity are all optional:
-- omit to keep the shape's current value.
methods.dev_retouch_update_shape = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_retouch_update_shape: op required") end
  local instance = tonumber(p.instance) or 0
  local formid = tonumber(p.formid)
  if formid == nil then error("dev_retouch_update_shape: formid required") end
  local target = p.target
  if type(target) ~= "table" or tonumber(target.x) == nil or tonumber(target.y) == nil then
    error("dev_retouch_update_shape: target {x,y} required")
  end
  local source = p.source
  if type(source) ~= "table" or tonumber(source.x) == nil or tonumber(source.y) == nil then
    error("dev_retouch_update_shape: source {x,y} required")
  end
  local radius = tonumber(p.radius)
  if radius == nil then error("dev_retouch_update_shape: radius required") end
  local feather = tonumber(p.feather) or 0.0
  local algorithm = p.algorithm -- nil -> C keeps the shape's current algorithm
  local scale = tonumber(p.wavelet_scale) -- nil -> C keeps the shape's current scale
  local opacity = tonumber(p.opacity) -- nil -> C leaves opacity untouched
  return dt.develop.retouch_update_shape(op, instance, formid,
    tonumber(target.x), tonumber(target.y), radius, feather,
    tonumber(source.x), tonumber(source.y), algorithm, scale, opacity)
end

methods.dev_retouch_delete_shape = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_retouch_delete_shape: op required") end
  local instance = tonumber(p.instance) or 0
  local formid = tonumber(p.formid)
  if formid == nil then error("dev_retouch_delete_shape: formid required") end
  return dt.develop.retouch_delete_shape(op, instance, formid)
end

methods.dev_retouch_list_shapes = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_retouch_list_shapes: op required") end
  local instance = tonumber(p.instance) or 0
  return dt.develop.retouch_list_shapes(op, instance)
end

-- Rollback helper for mask_object (T2.3, ACCEPTANCE T2.3-S2 / PLAN.md gap
-- "no orphan instance"): removes the most-recently-created multi-instance of
-- `op` via the SAME "new instance" GUI action darktable's own module header
-- button uses -- dt.gui.action("iop/"..op, -1, "instance", "delete"), which
-- dt_action_process resolves to dt_iop_module_t* and dispatches to
-- _gui_delete_callback -> dt_dev_module_remove (src/develop/imageop.c:480,
-- :4074 DT_ACTION_EFFECT_DELETE case). No C change needed -- this reuses the
-- existing action-system passthrough darktable.gui.action() already exposes
-- (same mechanism as the T0.3 spike's `nudge`).
--
-- `-1` asks dt_action_process to count instances of this op from the TAIL of
-- dev->iop (src/gui/accelerators.c _process_action: `instance<0` walks
-- g_list_last/g_list_previous). dt_dev_module_duplicate always inserts a new
-- instance immediately after the base (multi_priority 0) instance in
-- iop_order (src/develop/develop.c:3706 dt_ioppr_move_iop_after), so right
-- after mask_object's own add_instance(op) call the newest instance IS the
-- last (or only other) entry among this op's instances -- `-1` targets it
-- exactly. This is a narrow, single-purpose primitive for "undo the instance
-- I just created a moment ago", NOT a general "delete instance N" API: if
-- other instances of the same op were added/reordered by someone else
-- between add_instance and the rollback, `-1` may not resolve to the
-- intended instance. Requires >=2 instances of `op` to exist -- darktable's
-- own multi_show.close guard (imageop.c:_get_multi_show) -- which is always
-- true right after add_instance succeeded.
methods.dev_remove_last_instance = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_remove_last_instance: op required") end
  local action_path = "iop/" .. op
  -- IMPORTANT: the trailing `size` arg (any value != DT_READ_ACTION_ONLY,
  -- i.e. != -FLT_MAX) is NOT optional here. dt_action_process gates every
  -- state-changing effect behind DT_PERFORM_ACTION(move_size) (src/common/
  -- action.h: `(move_size) != DT_READ_ACTION_ONLY`); omitting it (as a bare
  -- 4-arg dt.gui.action call defaults move_size to DT_READ_ACTION_ONLY,
  -- src/lua/gui.c _action_cb) makes this call-and-return a harmless READ
  -- that resolves the target module and returns 0 WITHOUT invoking
  -- _gui_delete_callback -- confirmed empirically: it returns a valid
  -- (non-NaN) value and reports ok=true while leaving dev->iop completely
  -- unchanged. `1.0` below is the "perform it" trigger value, not a
  -- meaningful magnitude (the instance/delete effect ignores its size).
  local ok, ret = pcall(function()
    return dt.gui.action(action_path, -1, "instance", "delete", 1.0)
  end)
  if not ok then
    return { ok = false, op = op, error = "dt.gui.action raised: " .. tostring(ret) }
  end
  if ret ~= ret then -- NaN: dt_action_process's DT_ACTION_NOT_VALID sentinel
    return { ok = false, op = op,
             error = "dt.gui.action returned invalid (only one instance, or action path not found)" }
  end
  return { ok = true, op = op }
end

-- Read-only darkroom canvas zoom/pan for the main window and, when open on a
-- second monitor, the preview2 window. Returns the C result verbatim:
-- {main={...}, preview2={active=bool,...}} | {error=...}. Each viewport carries
-- a `region` {x,y,w,h} (top-left, normalized 0..1, clamped to [0,1]) = the crop
-- of the full image visible in that window; pass it straight into get_preview's
-- `region`. Raw zoom_x/zoom_y are center-relative (can be negative), NOT a
-- drop-in region.
methods.dev_get_viewport = function(p)
  return dt.develop.get_viewport()
end

-- Writable counterpart to dev_get_viewport (2026-07-31 set-viewport-design):
-- set an ABSOLUTE zoom_x/zoom_y/scale on "main" (dev->full) or "preview2".
-- Free zoom only (no fit/fill/1:1 snapping) -- the C side clamps `scale` to
-- darktable's own unconstrained-zoom bound and reports the clamp. This
-- wrapper does NOT do region math (aspect-expand, mode translation,
-- min_render_px_across) -- that lives in server.py, which reads
-- dev_get_viewport()'s processed_width/processed_height + current region/
-- scale to invert the region formula, then calls this with the resulting
-- zoom_x/zoom_y/scale. `wait_for_pipe` (default true) blocks (bounded by
-- `timeout_ms`, default 4000) until dev->full.pipe/dev->preview2.pipe has
-- actually reprocessed at the new zoom -- see dt.develop.set_viewport's own
-- long doc comment in src/lua/develop.c for why this is NOT the same
-- freshness check dev_preview uses. Returns the C result verbatim:
-- {ok, viewport, previous={zoom,zoom_label,closeup,zoom_x,zoom_y,scale},
-- applied={zoom_x,zoom_y,scale}, clamped=[...], pipe_ready, waited_ms} |
-- {error="viewport_not_active: ..."} | {error=...}.
methods.dev_set_viewport = function(p)
  p = p or {}
  local viewport = p.viewport or "main"
  local zoom_x = tonumber(p.zoom_x)
  local zoom_y = tonumber(p.zoom_y)
  local scale = tonumber(p.scale)
  if zoom_x == nil or zoom_y == nil or scale == nil then
    error("dev_set_viewport: zoom_x, zoom_y, scale required")
  end
  local wait_for_pipe = p.wait_for_pipe
  if wait_for_pipe == nil then wait_for_pipe = true end
  local timeout_ms = tonumber(p.timeout_ms)
  return dt.develop.set_viewport(viewport, zoom_x, zoom_y, scale,
    wait_for_pipe and true or false, timeout_ms)
end

-- Put a viewport's zoom/pan back exactly as dev_set_viewport (or
-- dev_get_viewport) reported it in `previous`/its own snapshot -- pass that
-- SAME table straight through as `previous` (no new state to invent, per
-- the set-viewport-design task's requirement 7). Same wait_for_pipe/
-- timeout_ms semantics as dev_set_viewport; no clamping (a state that was
-- once valid is restored as-is). Returns {ok, viewport, pipe_ready,
-- waited_ms} | {error="viewport_not_active: ..."} | {error=...}.
methods.dev_restore_viewport = function(p)
  p = p or {}
  local viewport = p.viewport or "main"
  local prev = p.previous
  if type(prev) ~= "table" then
    error("dev_restore_viewport: previous table required (zoom, closeup, zoom_x, zoom_y, scale)")
  end
  local zoom = tonumber(prev.zoom)
  local closeup = tonumber(prev.closeup)
  local zoom_x = tonumber(prev.zoom_x)
  local zoom_y = tonumber(prev.zoom_y)
  local scale = tonumber(prev.scale)
  if zoom == nil or closeup == nil or zoom_x == nil or zoom_y == nil or scale == nil then
    error("dev_restore_viewport: previous must have zoom, closeup, zoom_x, zoom_y, scale")
  end
  local wait_for_pipe = p.wait_for_pipe
  if wait_for_pipe == nil then wait_for_pipe = true end
  local timeout_ms = tonumber(p.timeout_ms)
  return dt.develop.restore_viewport(viewport, zoom, closeup, zoom_x, zoom_y, scale,
    wait_for_pipe and true or false, timeout_ms)
end

-- Bugreport (2026-07-25): a caller deriving a normalized point from
-- get_viewport()/get_preview() (the PROCESSED/display frame) and handing it
-- STRAIGHT to dev_retouch_add_shape/dev_add_path_mask (which store points in
-- the PIPE-INPUT/mask frame -- see dt.develop.backtransform_point's doc
-- comment in src/lua/develop.c) places the shape on the wrong part of the
-- image whenever orientation/crop/rotate/lens-correction is active -- these
-- two frames are NOT the same, confirmed via a real portrait photo where a
-- neck target landed on the chest even with a full, uncropped region. This
-- wrapper is the missing conversion step; len1/len2 are optional lengths
-- (e.g. radius/feather) converted alongside x/y in the same call.
methods.dev_backtransform_point = function(p)
  p = p or {}
  local x = tonumber(p.x)
  local y = tonumber(p.y)
  if x == nil or y == nil then error("dev_backtransform_point: x/y required") end
  local len1 = tonumber(p.len1)
  local len2 = tonumber(p.len2)
  return dt.develop.backtransform_point(x, y, len1, len2)
end

-- The INVERSE of dev_backtransform_point: mask-frame normalized point (what
-- dev_retouch_list_shapes/dev_get_mask report as stored geometry) -> processed/
-- display-frame normalized point (what get_preview/capture_viewport render).
-- Needed to plot an EXISTING shape on top of a captured render, or to check
-- that a written shape landed where it was aimed. len1/len2 are optional
-- lengths (radius/feather) in dt_masks' mindim-normalized convention, returned
-- normalized against the display frame WIDTH.
methods.dev_transform_point = function(p)
  p = p or {}
  local x = tonumber(p.x)
  local y = tonumber(p.y)
  if x == nil or y == nil then error("dev_transform_point: x/y required") end
  local len1 = tonumber(p.len1)
  local len2 = tonumber(p.len2)
  return dt.develop.transform_point(x, y, len1, len2)
end

-- T3.3 (promoted from darktable-mcp/spike/spike_methods.lua once the T3.1
-- go/no-go spike proved the underlying C binding, src/lua/develop.c
-- set_raster_source_cb): wire a downstream (consumer) module's blend to
-- consume the RASTER mask emitted by an upstream (source) module -- e.g. the
-- stock iop/rasterfile.c producer (T3.1, IOP_FLAGS_WRITE_RASTER) reading an
-- external PFM/PNG file. This is the ONE piece add_path_mask (T2.2, drawn
-- masks only) cannot do: a soft, per-pixel raster alpha instead of a hard
-- polygon boundary. opacity is 0..100 (percent, default 100.0) matching the
-- C binding's own blend_params->opacity scale -- NOT the 0..1 fraction
-- add_path_mask uses; mask_raster (T3.3, server.py) converts its 0..1
-- input before calling this. Returns the C result verbatim: {ok, consumer,
-- consumer_instance, source, source_instance, raster_mask_source,
-- raster_mask_instance, mask_mode, opacity} | {error=...}. The source
-- module must be EARLIER in the pixelpipe (lower iop_order) than the
-- consumer, or the C side returns an error.
methods.dev_set_raster_source = function(p)
  p = p or {}
  local cop = p.consumer_op or p.op
  if not cop or cop == "" then error("dev_set_raster_source: consumer_op required") end
  local sop = p.source_op
  if not sop or sop == "" then error("dev_set_raster_source: source_op required") end
  local cinst = tonumber(p.consumer_instance) or 0
  local sinst = tonumber(p.source_instance) or 0
  if p.opacity ~= nil then
    return dt.develop.set_raster_source(cop, cinst, sop, sinst, tonumber(p.opacity))
  end
  return dt.develop.set_raster_source(cop, cinst, sop, sinst)
end

-- Read-only enumeration of the drawn masks wired into ONE module instance's
-- blend group (mask_id/type/opacity/nb_points/name/operation/invert per
-- shape) -- the per-module half of get_module_mask; combine with
-- dev_get_blend_params (group-level opacity/mask_mode/blend_mode/invert) for
-- the full picture. Returns [] when the module has no drawn masks.
methods.dev_list_masks = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_list_masks: op required") end
  local instance = tonumber(p.instance) or 0
  return dt.develop.list_masks(op, instance)
end

-- Read-only enumeration of EVERY drawn mask shape in the current image,
-- independent of which module currently uses it -- so a caller can find a
-- shape drawn earlier (by hand in the GUI, or by another tool this session)
-- and attach it to a NEW module instance via dev_attach_mask, instead of
-- guessing a formid or redrawing it. Each entry also reports `used_by`
-- ({op, instance} pairs) so a caller can see who else already references it.
methods.dev_list_all_masks = function(p)
  return dt.develop.list_all_masks()
end

-- Read-only dump of ONE mask form's full point geometry (corner/ctrl1/ctrl2/
-- border/state per node for path/brush; single-entry descriptor for circle/
-- ellipse), plus its name -- so a caller can inspect/verify a mask's actual
-- shape (e.g. whether a node is a smooth or corner point: ctrl1/ctrl2 equal
-- to corner means corner, differing means smooth) without re-segmenting.
-- dt.develop.get_mask itself already existed in C (registered, unused until
-- 2026-07-31) -- this was the missing Lua-bridge wiring, not new C.
-- Points are in the same PIPE-INPUT/mask-frame convention as
-- add_path_mask's input -- see dt.develop.backtransform_point's doc comment
-- for why that differs from get_preview's display frame under crop/rotate/
-- orientation.
methods.dev_get_mask = function(p)
  p = p or {}
  local mask_id = tonumber(p.mask_id)
  if mask_id == nil then error("dev_get_mask: mask_id required") end
  return dt.develop.get_mask(mask_id)
end

-- Give a drawn mask a caller-chosen name instead of the auto-generated
-- "path #7"/"circle #3" (bugreport 2026-07-31: no way to tell masks apart
-- afterward except by formid, once several have been created in one
-- session). Returns {ok, mask_id, name} with name read back from the live
-- struct, not the requested string.
methods.dev_rename_mask = function(p)
  p = p or {}
  local mask_id = tonumber(p.mask_id)
  if mask_id == nil then error("dev_rename_mask: mask_id required") end
  local name = p.name
  if name == nil then error("dev_rename_mask: name required") end
  return dt.develop.rename_mask(mask_id, tostring(name))
end

-- Permanently delete a drawn mask shape, whether or not it's currently
-- attached to any module (2026-07-31 bugreport: no way to clean up an
-- orphan mask left over from experimentation -- detach_mask only unwires
-- it, still leaving it in dev->forms forever).
methods.dev_delete_mask = function(p)
  p = p or {}
  local mask_id = tonumber(p.mask_id)
  if mask_id == nil then error("dev_delete_mask: mask_id required") end
  return dt.develop.delete_mask(mask_id)
end

-- Wire an EXISTING drawn mask shape (formid, from dev_list_all_masks or the
-- return value of dev_add_path_mask/dev_mask_object/dev_retouch_add_shape)
-- into module (op, instance)'s blend group WITHOUT copying it -- the shape
-- stays one dev->forms entry, now referenced by this module's group in
-- ADDITION to whatever already referenced it (the same shape can back
-- several modules at once). `operation` (default "union") selects how this
-- shape combines with whatever else is already in the group: "union",
-- "intersection", "difference", "exclusion".
--
-- Also ORs DEVELOP_MASK_MASK|DEVELOP_MASK_ENABLED into blend_params->
-- mask_mode (bugreport 2026-07-27: wiring the group reference alone left
-- the mask inert -- dt_dev_pixelpipe's _piece_wants_blending gates the
-- actual blend on DEVELOP_MASK_ENABLED, so the shape existed but had no
-- visible effect until the pencil icon was clicked by hand). The returned
-- `mask_mode` is read back from the live struct after the write, so a
-- caller can verify it stuck instead of trusting the call silently.
methods.dev_attach_mask = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_attach_mask: op required") end
  local instance = tonumber(p.instance) or 0
  local formid = tonumber(p.formid)
  if formid == nil then error("dev_attach_mask: formid required") end
  local operation = p.operation
  return dt.develop.attach_mask(op, instance, formid, operation)
end

-- Unwire a drawn mask shape from module (op, instance)'s blend group WITHOUT
-- deleting the shape itself -- it stays in dev->forms and can be re-attached
-- (to this module or another) via dev_attach_mask. If this was the LAST shape
-- in the group, darktable's own dt_masks_form_remove cascades into deleting
-- the now-empty group, which resets EVERY module referencing it back to "no
-- mask" -- expected, mirrors the GUI's own "no masks" action exactly.
methods.dev_detach_mask = function(p)
  p = p or {}
  local op = p.op
  if not op or op == "" then error("dev_detach_mask: op required") end
  local instance = tonumber(p.instance) or 0
  local formid = tonumber(p.formid)
  if formid == nil then error("dev_detach_mask: formid required") end
  return dt.develop.detach_mask(op, instance, formid)
end

-- ---- Dispatch --------------------------------------------------------------

local function handle(req)
  local fn = methods[req.method]
  if not fn then
    return {id = req.id, error = "unknown method: " .. tostring(req.method)}
  end
  local ok, result_or_err = pcall(fn, req.params)
  if not ok then
    return {id = req.id, error = "handler raised: " .. tostring(result_or_err)}
  end
  return {id = req.id, result = result_or_err}
end

-- ---- File I/O --------------------------------------------------------------

local function cache_dir()
  local base = os.getenv("XDG_CACHE_HOME")
  if not base or base == "" then
    base = os.getenv("HOME") .. "/.cache"
  end
  return base .. "/darktable-mcp"
end

local function list_request_files(dir)
  local out = {}
  local p = io.popen('ls -1 "' .. dir .. '"/request-*.json 2>/dev/null')
  if not p then return out end
  for line in p:lines() do
    if not line:match("%.tmp$") then table.insert(out, line) end
  end
  p:close()
  return out
end

local function read_file(path)
  local f = io.open(path, "r")
  if not f then return nil end
  local content = f:read("*a")
  f:close()
  return content
end

local function write_file_atomic(path, content)
  local tmp = path .. ".tmp"
  local f = io.open(tmp, "w")
  if not f then return false end
  f:write(content)
  f:close()
  return os.rename(tmp, path)
end

local function scan_dir(dir)
  for _, req_path in ipairs(list_request_files(dir)) do
    local content = read_file(req_path)
    if content then
      local ok, req = pcall(json.decode, content)
      if ok and type(req) == "table" and type(req.id) == "string" and req.id:match("^[%w%-_]+$") then
        local resp = handle(req)
        local resp_path = dir .. "/response-" .. req.id .. ".json"
        write_file_atomic(resp_path, json.encode(resp))
      end
      os.remove(req_path)
    end
  end
end

local function sweep_stale(dir, max_age_seconds)
  -- Simple approach: shell out. find prints paths older than N minutes;
  -- we use seconds via -mmin with arithmetic. For the MVP, 60s = 1min.
  local minutes = math.max(1, math.floor(max_age_seconds / 60))
  os.execute(string.format(
    'find "%s" -maxdepth 1 -name "request-*.json" -mmin +%d -delete 2>/dev/null',
    dir, minutes))
end

-- ---- Worker loop -----------------------------------------------------------

local function worker_loop()
  local dir = cache_dir()
  os.execute('mkdir -p "' .. dir .. '"')
  local tick = 0
  while true do
    scan_dir(dir)
    tick = tick + 1
    if tick % 100 == 0 then
      sweep_stale(dir, 60)
    end
    if dt.control and dt.control.sleep then
      dt.control.sleep(100)
    else
      dt.print_log("darktable-mcp: dt.control.sleep missing, worker exiting")
      return
    end
  end
end

-- ---- Entry point -----------------------------------------------------------

dt.print_log("darktable-mcp bridge: ready")
if dt.control and dt.control.dispatch then
  dt.control.dispatch(worker_loop)
end

-- ---- Test exports (used by tests/lua/test_dispatcher.lua) ------------------
return {
  handle = handle,
  scan_dir = scan_dir,
  methods = methods,
  json = json,
}
