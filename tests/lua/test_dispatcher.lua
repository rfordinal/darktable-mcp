-- Lua dispatcher unit tests. Stubs the darktable `dt` global with a fake
-- database and verifies the method registry + scan_dir behavior in isolation.
--
-- Run via: lua tests/lua/test_dispatcher.lua
-- Exits 0 on success, non-zero on first failure.

-- ---- Test harness ----------------------------------------------------------
local failures = {}
local function assertEq(actual, expected, label)
  if actual ~= expected then
    table.insert(failures, string.format("%s: expected %s, got %s",
      label, tostring(expected), tostring(actual)))
  end
end
local function assertTrue(cond, label)
  if not cond then
    table.insert(failures, string.format("%s: expected truthy, got falsy", label))
  end
end

-- ---- Stub dt table ---------------------------------------------------------
-- Use ids that DO NOT overlap the iteration array's integer indexes (1..N),
-- so dt.database[id] always resolves via __index without colliding with
-- the iteration storage.
local images_by_id = {
  [101] = {id = 101, filename = "DSC_0001.NEF", path = "/photos", rating = 5,
           sidecar = "/photos/DSC_0001.NEF.xmp"},
  [102] = {id = 102, filename = "DSC_0002.NEF", path = "/photos", rating = 3},
  [103] = {id = 103, filename = "OTHER.NEF",   path = "/photos", rating = 4},
}
-- attach_tag/detach_tag mutate the tag's own array part (see stub_tags
-- below), mirroring the real image:attach_tag(tag)/image:detach_tag(tag)
-- Lua API used by methods.tag_photo.
for _, img in pairs(images_by_id) do
  img.attach_tag = function(self, tag)
    for _, v in ipairs(tag) do if v == self.id then return end end
    table.insert(tag, self.id)
  end
  img.detach_tag = function(self, tag)
    for i, v in ipairs(tag) do
      if v == self.id then table.remove(tag, i); return end
    end
  end
end
local iter_list = {}
for _, img in pairs(images_by_id) do table.insert(iter_list, img) end
local stub_db = setmetatable(iter_list, {
  __index = function(_, k) return images_by_id[k] end,
})
-- Real darktable exposes dt.database.get_image(id) as the by-real-id lookup
-- (dt.database[n] is positional/OFFSET, a different thing entirely -- see
-- src-dt/src/lua/database.c:database_numindex vs database_get_image).
stub_db.get_image = function(id) return images_by_id[id] end

-- ---- Stub dt.tags -----------------------------------------------------
-- A tag object's array part holds the ids of images it's attached to, so
-- both `#tag` (count) and `tag:get_tagged_images()` work off plain ipairs.
local tags_by_name = {}
local tags_array = {}
local function make_tag(name)
  local tag = {name = name}
  tag.get_tagged_images = function(self)
    local imgs = {}
    for _, imgid in ipairs(self) do table.insert(imgs, images_by_id[imgid]) end
    return imgs
  end
  return tag
end
local stub_tags = setmetatable({}, {
  __index = function(_, k)
    if k == "find" then
      return function(name) return tags_by_name[name] end
    elseif k == "create" then
      return function(name)
        local tag = make_tag(name)
        tags_by_name[name] = tag
        table.insert(tags_array, tag)
        return tag
      end
    else
      return tags_array[k]
    end
  end,
})

local dt_log = {}
local stub_dt = {
  database = stub_db,
  -- dt.collection is the currently-open lighttable view; these tests don't
  -- exercise the collection-vs-library distinction itself (that's a
  -- darktable-core behavior, not plugin logic), so point it at the same
  -- fixture data as dt.database.
  collection = stub_db,
  tags = stub_tags,
  print_log = function(msg) table.insert(dt_log, msg) end,
  control = {
    dispatch = function(_) end,
    sleep = function(_) end,
  },
}
local gui_view_state = {name = "lighttable"}
stub_dt.gui = {
  -- Minimal stub: empty action_images means the selection never "resolves"
  -- (see open_darkroom's diagnostic pre-check), so most tests hit the
  -- early-return diagnostic branch without needing to simulate an actual
  -- lighttable<->darkroom view switch. current_view is stateful (records
  -- the last view "switched" to) so the same-view-is-a-pipeline-no-op bug
  -- (bugreport 2026-07-25) can actually be exercised: bouncing through
  -- lighttable must be visible as a real state change, not a fixed stub.
  selection = function(sel) return sel or {} end,
  action_images = {},
  current_view = function(target)
    if target then gui_view_state = {name = target.name} end
    return gui_view_state
  end,
  views = {
    darkroom = {name = "darkroom"},
    lighttable = {name = "lighttable"},
  },
}

-- ---- Stub dt.develop --------------------------------------------------
-- Records every call so tests can assert exactly what the Lua wrapper
-- methods (dev_retouch_*) forwarded to the (real, C-only, unstubbable) core
-- binding -- these tests lock in the bridge-layer arg validation/defaults
-- contract, not the C logic itself (see PLAN.md's retouch section).
local develop_calls = {}
stub_dt.develop = {
  retouch_add_shape = function(...)
    table.insert(develop_calls, {name = "retouch_add_shape", args = {...}})
    return {ok = true, formid = 42, algorithm = "heal", wavelet_scale = 2}
  end,
  retouch_update_shape = function(...)
    table.insert(develop_calls, {name = "retouch_update_shape", args = {...}})
    return {ok = true, formid = 42, algorithm = "heal", wavelet_scale = 2, opacity = 1.0}
  end,
  retouch_delete_shape = function(...)
    table.insert(develop_calls, {name = "retouch_delete_shape", args = {...}})
    return {ok = true}
  end,
  retouch_list_shapes = function(...)
    table.insert(develop_calls, {name = "retouch_list_shapes", args = {...}})
    return {module = "retouch", instance = 0, shapes = {}}
  end,
  -- Identity stub (no orientation/crop/lens-correction simulated): real C
  -- semantics are display-frame-normalized -> mask/pipe-input-frame
  -- normalized (2026-07-25 bugreport fix) -- only the bridge-layer
  -- arg-forwarding contract is under test here, not the transform math
  -- itself (that can only be verified against a real darktable process).
  backtransform_point = function(x, y, len1, len2)
    table.insert(develop_calls, {name = "backtransform_point", args = {x, y, len1, len2}})
    local out = {x = x, y = y}
    if len1 ~= nil then out.len1 = len1 end
    if len2 ~= nil then out.len2 = len2 end
    return out
  end,
  -- Identity stub for the opposite direction (mask/pipe-input frame ->
  -- processed/display frame), used when reading shapes back for verification
  -- or overlay drawing. Same caveat as above: only arg forwarding is tested.
  transform_point = function(x, y, len1, len2)
    table.insert(develop_calls, {name = "transform_point", args = {x, y, len1, len2}})
    local out = {x = x, y = y}
    if len1 ~= nil then out.len1 = len1 end
    if len2 ~= nil then out.len2 = len2 end
    return out
  end,
  -- Writable viewport zoom/pan (2026-07-31 set-viewport-design). Only the
  -- bridge-layer arg-forwarding/defaulting contract is under test here (the
  -- real dt_dev_zoom_move/clamp/wait-for-pipe semantics can only be verified
  -- against a live darktable process) -- canned response mirrors the C
  -- binding's documented shape.
  set_viewport = function(viewport, zoom_x, zoom_y, scale, wait_for_pipe, timeout_ms)
    table.insert(develop_calls, {name = "set_viewport",
      args = {viewport, zoom_x, zoom_y, scale, wait_for_pipe, timeout_ms}})
    return {
      ok = true, viewport = viewport,
      previous = {zoom = 3, zoom_label = "free", closeup = 0,
                  zoom_x = -0.02, zoom_y = 0.05, scale = 0.4},
      applied = {zoom_x = zoom_x, zoom_y = zoom_y, scale = scale},
      clamped = {}, pipe_ready = true, waited_ms = 15,
    }
  end,
  restore_viewport = function(viewport, zoom, closeup, zoom_x, zoom_y, scale, wait_for_pipe, timeout_ms)
    table.insert(develop_calls, {name = "restore_viewport",
      args = {viewport, zoom, closeup, zoom_x, zoom_y, scale, wait_for_pipe, timeout_ms}})
    return {ok = true, viewport = viewport, pipe_ready = true, waited_ms = 8}
  end,
  -- capture_viewport/get_preview fix (2026-07-31 set-viewport-design
  -- follow-up): preview(max_w, max_h[, x, y, w, h][, viewport]) -- viewport
  -- picks dev->full.pipe/dev->preview2.pipe instead of the default
  -- dev->preview_pipe. Canned response mirrors the real C binding's shape
  -- (viewport_source echoes back what was actually used).
  preview = function(max_w, max_h, x, y, w, h, viewport)
    table.insert(develop_calls, {name = "preview", args = {max_w, max_h, x, y, w, h, viewport}})
    return {
      status = "ok", path = "/tmp/preview.png", width = 640, height = 480,
      frame_width = 640, frame_height = 480,
      viewport_source = viewport or "preview_pipe",
    }
  end,
  -- LUT tooling (2026-07-26): reads an arbitrary core conf key. Stub returns
  -- a canned lut3d root path for the one key the tests exercise, "" for
  -- anything else (matches the real C binding's never-nil contract).
  get_conf_string = function(key)
    table.insert(develop_calls, {name = "get_conf_string", args = {key}})
    if key == "plugins/darkroom/lut3d/def_path" then
      return "/tmp/luts"
    end
    return ""
  end,
  -- Blend opacity control (2026-07-26): lut3d has no "amount" of its own, so
  -- intensity control goes through blend_params instead of module params.
  get_blend_params = function(op, instance)
    table.insert(develop_calls, {name = "get_blend_params", args = {op, instance}})
    return {op = op, instance = instance, opacity = 100.0, mask_mode = 0, blend_mode = 3}
  end,
  set_blend_params = function(op, instance, fields)
    table.insert(develop_calls, {name = "set_blend_params", args = {op, instance, fields}})
    return {
      ok = true, op = op, instance = instance,
      opacity = fields.opacity or 100.0,
      mask_mode = fields.enable_uniform_blend and 1 or 0,
      blend_mode = fields.blend_mode or 3,
      invert = fields.invert or false,
    }
  end,
  -- Mask group management (2026-07-27): attach/detach an EXISTING shape to a
  -- module's blend group without copying it, plus global/per-module listing.
  add_instance = function(op, fields)
    table.insert(develop_calls, {name = "add_instance", args = {op, fields}})
    local result = {ok = true, op = op, instance = 1, base_instance = 0, multi_name = ""}
    if fields ~= nil then
      result.fields_applied = {ok = true, applied = fields, clamped = {}, unknown_fields = {}}
    end
    return result
  end,
  list_masks = function(op, instance)
    table.insert(develop_calls, {name = "list_masks", args = {op, instance}})
    return {
      {mask_id = 42, type = "circle", opacity = 1.0, nb_points = 1, name = "circle 1",
       module = op, instance = instance, operation = "union", invert = false},
    }
  end,
  list_all_masks = function()
    table.insert(develop_calls, {name = "list_all_masks", args = {}})
    return {
      {formid = 42, name = "circle 1", type = "circle", used_by = {{op = "retouch", instance = 0}}},
    }
  end,
  attach_mask = function(op, instance, formid, operation)
    table.insert(develop_calls, {name = "attach_mask", args = {op, instance, formid, operation}})
    -- mask_mode=3 = DEVELOP_MASK_ENABLED|DEVELOP_MASK_MASK, matching the real
    -- C binding's post-2026-07-27-fix behavior (see attach_mask_cb).
    return {ok = true, op = op, instance = instance, formid = formid, operation = operation or "union", mask_mode = 3}
  end,
  detach_mask = function(op, instance, formid)
    table.insert(develop_calls, {name = "detach_mask", args = {op, instance, formid}})
    return {ok = true, op = op, instance = instance, formid = formid}
  end,
  get_mask = function(mask_id)
    table.insert(develop_calls, {name = "get_mask", args = {mask_id}})
    return {
      mask_id = mask_id, type = "path", name = "path #1",
      points = {{corner = {0.1, 0.2}, ctrl1 = {0.1, 0.2}, ctrl2 = {0.1, 0.2}, border = {0.02, 0.02}, state = 1}},
    }
  end,
  rename_mask = function(mask_id, name)
    table.insert(develop_calls, {name = "rename_mask", args = {mask_id, name}})
    return {ok = true, mask_id = mask_id, name = name}
  end,
  delete_mask = function(mask_id)
    table.insert(develop_calls, {name = "delete_mask", args = {mask_id}})
    return {ok = true, mask_id = mask_id}
  end,
  current_image = function()
    table.insert(develop_calls, {name = "current_image", args = {}})
    return {has_image = true, id = 101, path = "/photos/DSC_0001.NEF", filename = "DSC_0001.NEF"}
  end,
}

-- Make `require("darktable")` return our stub by pre-populating package.loaded.
package.loaded.darktable = stub_dt

-- ---- Load the plugin -------------------------------------------------------
package.path = package.path .. ";./darktable_mcp/lua/?.lua"
local internals = require("darktable_mcp")

-- ---- Verify the plugin announces itself on load ----------------------------
do
  assertEq(#dt_log, 1, "ready message logged on load")
  assertTrue(string.find(dt_log[1] or "", "ready"),
    "ready message contains the word 'ready'")
end

-- ---- methods.view_photos ---------------------------------------------------
do
  local result = internals.methods.view_photos({rating_min = 4, limit = 10})
  assertEq(#result, 2, "view_photos rating_min=4 returns 2 images")
  -- Order-independent: collect ratings, both must be >= 4, and the SET
  -- of ids must be {"101","103"} (the only images with rating >= 4).
  local ids_seen = {}
  for _, img in ipairs(result) do
    assertTrue(img.rating >= 4, "view_photos rating_min=4 image rating >= 4")
    ids_seen[img.id] = true
  end
  assertTrue(ids_seen["101"], "view_photos rating_min=4 includes id 101")
  assertTrue(ids_seen["103"], "view_photos rating_min=4 includes id 103")
end

do
  local result = internals.methods.view_photos({filter = "OTHER", limit = 10})
  assertEq(#result, 1, "view_photos filter=OTHER returns 1 image")
  assertEq(result[1].filename, "OTHER.NEF", "view_photos filter result filename")
  -- view_photos must return the absolute file path (dir + filename), not
  -- just the directory. Otherwise it doesn't compose with export_images,
  -- which takes file paths in `photo_ids`.
  assertEq(result[1].path, "/photos/OTHER.NEF",
    "view_photos returns absolute file path")
end

do
  local result = internals.methods.view_photos({limit = 2})
  assertEq(#result, 2, "view_photos limit=2 caps at 2")
end

-- ---- methods.view_photos: trailing-slash dir handling ---------------------
do
  -- Some darktable backends include a trailing slash on image.path; some
  -- don't. The plugin must normalize both into a single-slash full path.
  local trail_img = {id = 999, filename = "TRAIL.NEF", path = "/photos/", rating = 0}
  images_by_id[999] = trail_img
  table.insert(iter_list, trail_img)
  local result = internals.methods.view_photos({filter = "TRAIL", limit = 10})
  assertEq(#result, 1, "view_photos trailing-slash filter returns 1 image")
  assertEq(result[1].path, "/photos/TRAIL.NEF",
    "view_photos collapses double slash from trailing-dir input")
  -- Cleanup so later tests that count totals don't trip.
  images_by_id[999] = nil
  table.remove(iter_list)
end

-- ---- methods.view_photos: scope defaults to the open collection, not the
-- whole library (bug: view_photos was scanning dt.database unconditionally
-- and returning photos outside whatever collection/filter was open) --------
do
  -- Collection narrower than the library: only image 103.
  stub_dt.collection = setmetatable({images_by_id[103]}, {
    __index = function(_, k) return images_by_id[k] end,
  })
  local scoped = internals.methods.view_photos({limit = 10})
  assertEq(#scoped, 1, "view_photos default scope only sees the open collection")
  assertEq(scoped[1].id, "103", "view_photos default scope returns the collection's image")

  local whole = internals.methods.view_photos({limit = 10, scope = "library"})
  assertEq(#whole, 3, "view_photos scope=library ignores the open collection")

  stub_dt.collection = stub_db
end

-- ---- methods.rate_photos ---------------------------------------------------
do
  local result = internals.methods.rate_photos({photo_ids = {"101", "102"}, rating = 1})
  assertEq(result.updated, 2, "rate_photos updated count")
  assertEq(images_by_id[101].rating, 1, "rate_photos changed image 101 rating")
  assertEq(images_by_id[102].rating, 1, "rate_photos changed image 102 rating")
end

-- ---- methods.open_darkroom: real-id lookup (bugreport 2026-07-24: image ids
-- from view_photos/list_photos_in_collection are real database ids, NOT
-- positions -- dt.database[id] is a positional OFFSET lookup and returning
-- the wrong/no image for any id past the library's row count or past any
-- id gap; dt.database.get_image(id) is the actual by-id lookup) ------------
do
  local result = internals.methods.open_darkroom({image_id = "103"})
  assertTrue(result.diagnostic ~= nil,
    "open_darkroom (stubbed gui) hits the no-view-switch diagnostic branch")
  assertEq(result.requested_image_id, 103,
    "open_darkroom resolves real image id 103 via get_image, not a position")
end

do
  local ok, err = pcall(internals.methods.open_darkroom, {image_id = "999999"})
  assertTrue(not ok, "open_darkroom errors for an id with no matching image")
  assertTrue(string.find(err or "", "image not found") ~= nil,
    "open_darkroom error names the missing id")
end

-- ---- methods.open_darkroom: bugreport 2026-07-25 -- switching
-- current_view to darkroom while ALREADY in darkroom is a pipeline no-op
-- (views/view.c only calls enter()/dt_dev_load_image() on a real
-- old_view != new_view transition), so opening a second image right after
-- the first must bounce through lighttable to force a genuine reload. -----
do
  -- Selection resolves cleanly this time (unlike the diagnostic-branch
  -- tests above) so execution reaches the actual view-switch logic.
  stub_dt.gui.current_view(stub_dt.gui.views.darkroom)
  stub_dt.gui.action_images = {images_by_id[103]}

  local result = internals.methods.open_darkroom({image_id = "103"})
  assertTrue(result.bounced_through_lighttable,
    "open_darkroom bounces through lighttable when already in darkroom")
  assertEq(result.view, "darkroom",
    "open_darkroom ends back in darkroom after the forced bounce")

  stub_dt.gui.action_images = {}
end

do
  -- Control case: NOT already in darkroom (coming from lighttable) should
  -- NOT bounce -- the real view.c transition already fires enter() on its
  -- own, so a bounce here would just be a pointless extra round trip.
  stub_dt.gui.current_view(stub_dt.gui.views.lighttable)
  stub_dt.gui.action_images = {images_by_id[103]}

  local result = internals.methods.open_darkroom({image_id = "103"})
  assertTrue(not result.bounced_through_lighttable,
    "open_darkroom does not bounce when coming from lighttable")
  assertEq(result.view, "darkroom", "open_darkroom still ends in darkroom")

  stub_dt.gui.action_images = {}
end

-- ---- methods.dev_retouch_add_shape / dev_retouch_delete_shape /
-- dev_retouch_list_shapes: bridge-layer arg validation/defaults/forwarding
-- contract (the actual shape-creation C logic lives in src/lua/develop.c
-- and can only be verified against a real darktable process). ------------
do
  develop_calls = {}
  local result = internals.methods.dev_retouch_add_shape({
    op = "retouch",
    algorithm = "heal",
    target = {x = 0.4, y = 0.3},
    source = {x = 0.35, y = 0.3},
    radius = 0.02,
  })
  assertEq(result.formid, 42, "dev_retouch_add_shape forwards the C result")
  assertEq(#develop_calls, 1, "dev_retouch_add_shape calls dt.develop.retouch_add_shape once")
  local call = develop_calls[1]
  assertEq(call.name, "retouch_add_shape", "correct C function called")
  assertEq(call.args[1], "retouch", "op forwarded")
  assertEq(call.args[2], 0, "instance defaults to 0")
  assertEq(call.args[3], "heal", "algorithm forwarded")
  assertEq(call.args[4], 0.4, "target.x forwarded")
  assertEq(call.args[5], 0.3, "target.y forwarded")
  assertEq(call.args[6], 0.02, "radius forwarded")
  assertEq(call.args[7], 0.0, "feather defaults to 0.0")
  assertEq(call.args[8], 0.35, "source.x forwarded")
  assertEq(call.args[9], 0.3, "source.y forwarded")
  assertEq(call.args[10], nil, "wavelet_scale omitted -> nil (C default)")
  assertEq(call.args[11], 1.0, "opacity defaults to 1.0")
end

do
  develop_calls = {}
  internals.methods.dev_retouch_add_shape({
    op = "retouch",
    instance = 1,
    algorithm = "clone",
    target = {x = 0.5, y = 0.5},
    source = {x = 0.6, y = 0.6},
    radius = 0.03,
    feather = 0.1,
    wavelet_scale = 3,
    opacity = 0.8,
  })
  local call = develop_calls[1]
  assertEq(call.args[2], 1, "instance forwarded when given")
  assertEq(call.args[7], 0.1, "feather forwarded when given")
  assertEq(call.args[10], 3, "wavelet_scale forwarded when given")
  assertEq(call.args[11], 0.8, "opacity forwarded when given")
end

do
  local ok, err = pcall(internals.methods.dev_retouch_add_shape, {
    op = "retouch", algorithm = "heal", target = {x = 0.4, y = 0.3}, radius = 0.02,
  })
  assertTrue(not ok, "dev_retouch_add_shape errors without a source point")
  assertTrue(string.find(err or "", "source") ~= nil, "error mentions source")
end

do
  -- "path" used to be unsupported (this test predates the path batch); now
  -- only genuinely unsupported types (e.g. "brush") are still rejected.
  local ok, err = pcall(internals.methods.dev_retouch_add_shape, {
    op = "retouch", algorithm = "heal", shape_type = "brush",
    target = {x = 0.4, y = 0.3}, source = {x = 0.35, y = 0.3}, radius = 0.02,
  })
  assertTrue(not ok, "dev_retouch_add_shape errors on unsupported shape_type")
  assertTrue(string.find(err or "", "circle") ~= nil, "error mentions circle")
end

do
  -- 2026-07-31 path batch: shape_type="path" requires >=3 points, and
  -- target/radius are NOT required (a polygon has no single center/radius).
  local ok, err = pcall(internals.methods.dev_retouch_add_shape, {
    op = "retouch", algorithm = "heal", shape_type = "path",
    source = {x = 0.35, y = 0.3},
  })
  assertTrue(not ok, "dev_retouch_add_shape errors without points for shape_type='path'")
  assertTrue(string.find(err or "", "points") ~= nil, "error mentions points")
end

do
  develop_calls = {}
  internals.methods.dev_retouch_add_shape({
    op = "retouch", algorithm = "heal", shape_type = "path",
    source = {x = 0.35, y = 0.3},
    points = {{x = 0.1, y = 0.1}, {x = 0.2, y = 0.1}, {x = 0.15, y = 0.2}},
    smooth = false,
  })
  local call = develop_calls[1]
  assertEq(call.args[19], "path", "shape_type forwarded")
  local pts = call.args[22]
  assertEq(#pts, 3, "points table forwarded with 3 nodes")
  assertEq(pts[1].x, 0.1, "point 1 x forwarded")
  assertEq(call.args[23], false, "smooth forwarded")
end

do
  -- 2026-07-31 ellipse batch: shape_type="ellipse" forwards radius_b/rotation
  -- as the new trailing C args (positions 19-21).
  develop_calls = {}
  internals.methods.dev_retouch_add_shape({
    op = "retouch", algorithm = "heal", shape_type = "ellipse",
    target = {x = 0.4, y = 0.3}, source = {x = 0.35, y = 0.3},
    radius = 0.04, radius_b = 0.02, rotation = 30,
  })
  local call = develop_calls[1]
  assertEq(call.args[19], "ellipse", "shape_type forwarded")
  assertEq(call.args[20], 0.02, "radius_b forwarded")
  assertEq(call.args[21], 30, "rotation forwarded")
end

do
  -- shape_type="ellipse" without radius_b: still succeeds (C defaults
  -- radius_b to radius), Lua does not inject a value itself.
  develop_calls = {}
  internals.methods.dev_retouch_add_shape({
    op = "retouch", algorithm = "heal", shape_type = "ellipse",
    target = {x = 0.4, y = 0.3}, source = {x = 0.35, y = 0.3}, radius = 0.04,
  })
  local call = develop_calls[1]
  assertEq(call.args[19], "ellipse", "shape_type forwarded")
  assertEq(call.args[20], nil, "radius_b omitted -> nil (C defaults to radius)")
  assertEq(call.args[21], nil, "rotation omitted -> nil (C defaults to 0)")
end

-- ---- methods.dev_retouch_update_shape: bridge-layer arg validation/
-- defaults/forwarding for in-place move/resize (2026-07-25 viewport-relative
-- retouch design). algorithm/wavelet_scale/opacity are all optional (nil ->
-- C keeps the shape's current value); target/source/radius/feather are
-- always resent in full (a "move", not a partial field patch).
do
  develop_calls = {}
  local result = internals.methods.dev_retouch_update_shape({
    op = "retouch",
    formid = 42,
    target = {x = 0.5, y = 0.4},
    source = {x = 0.45, y = 0.4},
    radius = 0.03,
  })
  assertEq(result.formid, 42, "dev_retouch_update_shape forwards the C result")
  assertEq(#develop_calls, 1, "dev_retouch_update_shape calls dt.develop.retouch_update_shape once")
  local call = develop_calls[1]
  assertEq(call.name, "retouch_update_shape", "correct C function called")
  assertEq(call.args[1], "retouch", "op forwarded")
  assertEq(call.args[2], 0, "instance defaults to 0")
  assertEq(call.args[3], 42, "formid forwarded")
  assertEq(call.args[4], 0.5, "target.x forwarded")
  assertEq(call.args[5], 0.4, "target.y forwarded")
  assertEq(call.args[6], 0.03, "radius forwarded")
  assertEq(call.args[7], 0.0, "feather defaults to 0.0")
  assertEq(call.args[8], 0.45, "source.x forwarded")
  assertEq(call.args[9], 0.4, "source.y forwarded")
  assertEq(call.args[10], nil, "algorithm omitted -> nil (C keeps current)")
  assertEq(call.args[11], nil, "wavelet_scale omitted -> nil (C keeps current)")
  assertEq(call.args[12], nil, "opacity omitted -> nil (C leaves it untouched)")
end

do
  develop_calls = {}
  internals.methods.dev_retouch_update_shape({
    op = "retouch",
    instance = 1,
    formid = 7,
    target = {x = 0.5, y = 0.5},
    source = {x = 0.6, y = 0.6},
    radius = 0.04,
    feather = 0.05,
    algorithm = "clone",
    wavelet_scale = 2,
    opacity = 0.7,
  })
  local call = develop_calls[1]
  assertEq(call.args[2], 1, "instance forwarded when given")
  assertEq(call.args[7], 0.05, "feather forwarded when given")
  assertEq(call.args[10], "clone", "algorithm forwarded when given")
  assertEq(call.args[11], 2, "wavelet_scale forwarded when given")
  assertEq(call.args[12], 0.7, "opacity forwarded when given")
end

do
  local ok, err = pcall(internals.methods.dev_retouch_update_shape, {
    op = "retouch", formid = 42, algorithm = "heal",
    target = {x = 0.4, y = 0.3}, radius = 0.02,
  })
  assertTrue(not ok, "dev_retouch_update_shape errors without a source point when algorithm='heal'")
  assertTrue(string.find(err or "", "source") ~= nil, "error mentions source")
end

do
  -- algorithm omitted entirely (keep the shape's current one) AND source
  -- omitted: the bridge no longer knows whether the shape's current
  -- algorithm needs a source, so it must NOT error here -- only C (which
  -- knows the shape's live algorithm) can make that call. 2026-07-31 blur/
  -- fill batch: source is nil for blur/fill algorithms, so a bare "move"
  -- call must be allowed through without one.
  develop_calls = {}
  local result = internals.methods.dev_retouch_update_shape({
    op = "retouch", formid = 42, target = {x = 0.4, y = 0.3}, radius = 0.02,
  })
  assertEq(result.formid, 42, "dev_retouch_update_shape without algorithm/source reaches C")
  local call = develop_calls[1]
  assertEq(call.args[8], nil, "source_x omitted -> nil when algorithm is unspecified and no source given")
  assertEq(call.args[9], nil, "source_y omitted -> nil when algorithm is unspecified and no source given")
end

do
  -- 2026-07-31 ellipse step 2: radius_b/rotation forwarded as new trailing
  -- C args (positions 20-21).
  develop_calls = {}
  internals.methods.dev_retouch_update_shape({
    op = "retouch", formid = 42, target = {x = 0.4, y = 0.3}, radius = 0.02,
    radius_b = 0.03, rotation = 45,
  })
  local call = develop_calls[1]
  assertEq(call.args[20], 0.03, "radius_b forwarded")
  assertEq(call.args[21], 45, "rotation forwarded")
end

do
  -- 2026-07-31 path batch: `points` implies path-update mode -- target/
  -- radius are NOT required in this case (a polygon has no single
  -- center/radius), and points/smooth forward to positions 22-23.
  develop_calls = {}
  internals.methods.dev_retouch_update_shape({
    op = "retouch", formid = 42,
    points = {{x = 0.5, y = 0.5}, {x = 0.55, y = 0.5}, {x = 0.52, y = 0.55}},
    smooth = false,
  })
  local call = develop_calls[1]
  local pts = call.args[22]
  assertEq(#pts, 3, "points forwarded with 3 nodes")
  assertEq(call.args[23], false, "smooth forwarded")
end

do
  local ok, err = pcall(internals.methods.dev_retouch_update_shape, {
    op = "retouch", formid = 42, points = {{x = 0.1, y = 0.1}, {x = 0.2, y = 0.1}},
  })
  assertTrue(not ok, "dev_retouch_update_shape errors with fewer than 3 points")
  assertTrue(string.find(err or "", "points") ~= nil, "error mentions points")
end

do
  local ok, err = pcall(internals.methods.dev_retouch_update_shape, {
    op = "retouch", target = {x = 0.4, y = 0.3}, source = {x = 0.35, y = 0.3}, radius = 0.02,
  })
  assertTrue(not ok, "dev_retouch_update_shape errors without a formid")
  assertTrue(string.find(err or "", "formid") ~= nil, "error mentions formid")
end

do
  develop_calls = {}
  local result = internals.methods.dev_retouch_delete_shape({op = "retouch", formid = 42})
  assertTrue(result.ok, "dev_retouch_delete_shape forwards the C result")
  local call = develop_calls[1]
  assertEq(call.name, "retouch_delete_shape", "correct C function called")
  assertEq(call.args[1], "retouch", "op forwarded")
  assertEq(call.args[2], 0, "instance defaults to 0")
  assertEq(call.args[3], 42, "formid forwarded")
end

do
  develop_calls = {}
  local result = internals.methods.dev_retouch_list_shapes({op = "retouch"})
  assertEq(result.module, "retouch", "dev_retouch_list_shapes forwards the C result")
  local call = develop_calls[1]
  assertEq(call.name, "retouch_list_shapes", "correct C function called")
  assertEq(call.args[1], "retouch", "op forwarded")
  assertEq(call.args[2], 0, "instance defaults to 0")
end

-- ---- methods.dev_set_viewport / methods.dev_restore_viewport: bridge-layer
-- arg forwarding + defaulting for the writable zoom/pan control
-- (2026-07-31 set-viewport-design). Only the wrapper contract is under test
-- (defaults, required-arg validation, previous-table unpacking) -- the real
-- dt_dev_zoom_move/clamp/wait-for-pipe semantics need a live darktable
-- process, flagged as such in the delegated task report.
do
  develop_calls = {}
  local result = internals.methods.dev_set_viewport({zoom_x = -0.1, zoom_y = 0.05, scale = 2.5})
  assertTrue(result.ok, "dev_set_viewport forwards the C result")
  assertEq(result.viewport, "main", "result viewport echoed back")
  local call = develop_calls[1]
  assertEq(call.name, "set_viewport", "correct C function called")
  assertEq(call.args[1], "main", "viewport defaults to 'main'")
  assertEq(call.args[2], -0.1, "zoom_x forwarded")
  assertEq(call.args[3], 0.05, "zoom_y forwarded")
  assertEq(call.args[4], 2.5, "scale forwarded")
  assertEq(call.args[5], true, "wait_for_pipe defaults to true")
  assertEq(call.args[6], nil, "timeout_ms forwarded as nil when omitted (C applies its own default)")
end

do
  develop_calls = {}
  local result = internals.methods.dev_set_viewport({
    viewport = "preview2", zoom_x = 0.0, zoom_y = 0.0, scale = 1.0,
    wait_for_pipe = false, timeout_ms = 1000,
  })
  assertTrue(result.ok, "dev_set_viewport forwards the C result (preview2)")
  local call = develop_calls[1]
  assertEq(call.args[1], "preview2", "viewport forwarded when given")
  assertEq(call.args[5], false, "wait_for_pipe forwarded when explicitly false")
  assertEq(call.args[6], 1000, "timeout_ms forwarded when given")
end

do
  local ok, err = pcall(internals.methods.dev_set_viewport, {zoom_x = 0.1, zoom_y = 0.1})
  assertTrue(not ok, "dev_set_viewport errors without scale")
  assertTrue(string.find(err or "", "zoom_x, zoom_y, scale") ~= nil,
    "error names the missing required fields")
end

do
  develop_calls = {}
  local previous = {zoom = 0, zoom_label = "fit", closeup = 0, zoom_x = 0.0, zoom_y = 0.0, scale = 0.35}
  local result = internals.methods.dev_restore_viewport({previous = previous})
  assertTrue(result.ok, "dev_restore_viewport forwards the C result")
  local call = develop_calls[1]
  assertEq(call.name, "restore_viewport", "correct C function called")
  assertEq(call.args[1], "main", "viewport defaults to 'main'")
  assertEq(call.args[2], 0, "previous.zoom unpacked positionally")
  assertEq(call.args[3], 0, "previous.closeup unpacked positionally")
  assertEq(call.args[4], 0.0, "previous.zoom_x unpacked positionally")
  assertEq(call.args[5], 0.0, "previous.zoom_y unpacked positionally")
  assertEq(call.args[6], 0.35, "previous.scale unpacked positionally")
  assertEq(call.args[7], true, "wait_for_pipe defaults to true")
end

do
  develop_calls = {}
  local previous = {zoom = 3, zoom_label = "free", closeup = 1, zoom_x = -0.2, zoom_y = 0.3, scale = 4.0}
  internals.methods.dev_restore_viewport({
    viewport = "preview2", previous = previous, wait_for_pipe = false, timeout_ms = 2000,
  })
  local call = develop_calls[1]
  assertEq(call.args[1], "preview2", "viewport forwarded when given")
  assertEq(call.args[7], false, "wait_for_pipe forwarded when explicitly false")
  assertEq(call.args[8], 2000, "timeout_ms forwarded when given")
end

do
  local ok, err = pcall(internals.methods.dev_restore_viewport, {})
  assertTrue(not ok, "dev_restore_viewport errors without previous")
  assertTrue(string.find(err or "", "previous") ~= nil, "error mentions previous")
end

do
  local ok, err = pcall(internals.methods.dev_restore_viewport, {previous = {zoom = 0, closeup = 0}})
  assertTrue(not ok, "dev_restore_viewport errors when previous is missing fields")
  assertTrue(string.find(err or "", "zoom_x") ~= nil, "error names a missing field")
end

-- ---- methods.dev_preview: viewport= forwarding (2026-07-31 set-viewport-
-- design follow-up, the fix that makes capture_viewport('main') actually
-- reflect a prior set_viewport zoom instead of always reading the
-- fixed-resolution preview_pipe). Omitted viewport must still forward as
-- nil -- no behavior change for every pre-existing dev_preview caller.
do
  develop_calls = {}
  local result = internals.methods.dev_preview({max_w = 1024, max_h = 1024})
  assertEq(result.viewport_source, "preview_pipe", "no viewport -> stub echoes preview_pipe")
  local call = develop_calls[1]
  assertEq(call.name, "preview", "correct C function called")
  assertEq(call.args[1], 1024, "max_w forwarded")
  assertEq(call.args[2], 1024, "max_h forwarded")
  assertEq(call.args[3], nil, "region x forwarded as nil when omitted")
  assertEq(call.args[7], nil, "viewport forwarded as nil when omitted")
end

do
  develop_calls = {}
  local result = internals.methods.dev_preview({max_w = 1400, max_h = 1400, viewport = "main"})
  assertEq(result.viewport_source, "main", "viewport='main' forwarded and echoed")
  local call = develop_calls[1]
  assertEq(call.args[3], nil, "no region forwarded (capture_viewport's own fix: full.pipe's "
    .. "backbuf IS already the zoomed crop, passing region would double-crop)")
  assertEq(call.args[7], "main", "viewport forwarded")
end

do
  develop_calls = {}
  local result = internals.methods.dev_preview({
    max_w = 800, max_h = 800, viewport = "preview2",
    region = {x = 0.1, y = 0.2, w = 0.3, h = 0.4},
  })
  assertEq(result.viewport_source, "preview2", "viewport='preview2' forwarded and echoed")
  local call = develop_calls[1]
  assertEq(call.args[3], 0.1, "region.x forwarded when explicitly given together with viewport")
  assertEq(call.args[4], 0.2, "region.y forwarded")
  assertEq(call.args[5], 0.3, "region.w forwarded")
  assertEq(call.args[6], 0.4, "region.h forwarded")
  assertEq(call.args[7], "preview2", "viewport forwarded")
end

-- ---- methods.dev_backtransform_point: bridge-layer arg forwarding for the
-- display-frame -> mask-frame conversion (2026-07-25 coordinate-frame
-- bugreport fix). len1/len2 optional -- nil when omitted, forwarded when given.
do
  develop_calls = {}
  local result = internals.methods.dev_backtransform_point({x = 0.5, y = 0.3})
  assertEq(result.x, 0.5, "dev_backtransform_point forwards x (identity stub)")
  assertEq(result.y, 0.3, "dev_backtransform_point forwards y (identity stub)")
  assertEq(result.len1, nil, "len1 omitted when not given")
  local call = develop_calls[1]
  assertEq(call.name, "backtransform_point", "correct C function called")
  assertEq(call.args[1], 0.5, "x forwarded")
  assertEq(call.args[2], 0.3, "y forwarded")
  assertEq(call.args[3], nil, "len1 forwarded as nil when omitted")
  assertEq(call.args[4], nil, "len2 forwarded as nil when omitted")
end

do
  develop_calls = {}
  local result = internals.methods.dev_backtransform_point({x = 0.5, y = 0.3, len1 = 0.02, len2 = 0.01})
  assertEq(result.len1, 0.02, "len1 forwarded and returned")
  assertEq(result.len2, 0.01, "len2 forwarded and returned")
  local call = develop_calls[1]
  assertEq(call.args[3], 0.02, "len1 forwarded when given")
  assertEq(call.args[4], 0.01, "len2 forwarded when given")
end

do
  local ok, err = pcall(internals.methods.dev_backtransform_point, {y = 0.3})
  assertTrue(not ok, "dev_backtransform_point errors without x")
  assertTrue(string.find(err or "", "x/y") ~= nil, "error mentions x/y")
end

-- ---- methods.dev_transform_point: the mask-frame -> display-frame direction,
-- needed to verify or DRAW an existing shape against a captured render
-- (retouch_render_overlay, 2026-07-26). Must call the C `transform_point`, not
-- the backtransform -- mixing the two silently mirrors every coordinate.
do
  develop_calls = {}
  local result = internals.methods.dev_transform_point({x = 0.4, y = 0.6, len1 = 0.02})
  assertEq(result.x, 0.4, "dev_transform_point forwards x (identity stub)")
  assertEq(result.y, 0.6, "dev_transform_point forwards y (identity stub)")
  assertEq(result.len1, 0.02, "len1 forwarded and returned")
  assertEq(result.len2, nil, "len2 omitted when not given")
  local call = develop_calls[1]
  assertEq(call.name, "transform_point", "correct C function called (not backtransform)")
  assertEq(call.args[1], 0.4, "x forwarded")
  assertEq(call.args[3], 0.02, "len1 forwarded")
  assertEq(call.args[4], nil, "len2 forwarded as nil when omitted")
end

do
  local ok, err = pcall(internals.methods.dev_transform_point, {x = 0.4})
  assertTrue(not ok, "dev_transform_point errors without y")
  assertTrue(string.find(err or "", "x/y") ~= nil, "error mentions x/y")
end

-- ---- methods.dev_get_conf_string: LUT tooling (2026-07-26), needed to read
-- the lut3d module's configured root dir (plugins/darkroom/lut3d/def_path)
-- so list_luts scans the SAME directory as the darktable UI dropdown.
do
  develop_calls = {}
  local result = internals.methods.dev_get_conf_string({key = "plugins/darkroom/lut3d/def_path"})
  assertEq(result, "/tmp/luts", "dev_get_conf_string forwards the C return value")
  local call = develop_calls[1]
  assertEq(call.name, "get_conf_string", "correct C function called")
  assertEq(call.args[1], "plugins/darkroom/lut3d/def_path", "key forwarded")
end

do
  local ok, err = pcall(internals.methods.dev_get_conf_string, {})
  assertTrue(not ok, "dev_get_conf_string errors without key")
  assertTrue(string.find(err or "", "key") ~= nil, "error mentions key")
end

-- ---- methods.dev_get_blend_params / dev_set_blend_params: blend opacity
-- control (2026-07-26), needed for LUT intensity since lut3d has no "amount"
-- of its own -- blend_params is a separate flat struct from module params.
do
  develop_calls = {}
  local result = internals.methods.dev_get_blend_params({op = "lut3d", instance = 0})
  assertEq(result.opacity, 100.0, "dev_get_blend_params forwards opacity")
  local call = develop_calls[1]
  assertEq(call.name, "get_blend_params", "correct C function called")
  assertEq(call.args[1], "lut3d", "op forwarded")
  assertEq(call.args[2], 0, "instance forwarded")
end

do
  local ok, err = pcall(internals.methods.dev_get_blend_params, {})
  assertTrue(not ok, "dev_get_blend_params errors without op")
end

do
  develop_calls = {}
  local result = internals.methods.dev_set_blend_params({
    op = "lut3d", instance = 0,
    fields = {opacity = 35.0, enable_uniform_blend = true},
  })
  assertEq(result.ok, true, "dev_set_blend_params reports ok")
  assertEq(result.opacity, 35.0, "dev_set_blend_params forwards opacity")
  local call = develop_calls[1]
  assertEq(call.name, "set_blend_params", "correct C function called")
  assertEq(call.args[3].opacity, 35.0, "fields forwarded")
  assertEq(call.args[3].enable_uniform_blend, true, "enable_uniform_blend forwarded")
end

do
  local ok, err = pcall(internals.methods.dev_set_blend_params, {op = "lut3d", instance = 0})
  assertTrue(not ok, "dev_set_blend_params errors without fields")
  assertTrue(string.find(err or "", "fields") ~= nil, "error mentions fields")
end

-- ---- methods.dev_add_instance: fields={} initial-param overrides
-- (2026-07-27) -- exposure's compensate_exposure_bias/compensate_hilite_pres
-- must be settable on the NEW instance without a second history entry.
do
  develop_calls = {}
  local result = internals.methods.dev_add_instance({op = "exposure"})
  assertEq(result.instance, 1, "dev_add_instance without fields forwards result")
  local call = develop_calls[1]
  assertEq(call.name, "add_instance", "correct C function called")
  assertEq(call.args[1], "exposure", "op forwarded")
  assertTrue(call.args[2] == nil, "no fields table forwarded when fields omitted")
  assertTrue(result.fields_applied == nil, "no fields_applied when fields omitted")
end

do
  develop_calls = {}
  local result = internals.methods.dev_add_instance({
    op = "exposure",
    fields = {compensate_exposure_bias = false, compensate_hilite_pres = false},
  })
  local call = develop_calls[1]
  assertEq(call.args[2].compensate_exposure_bias, false, "fields forwarded to C call")
  assertTrue(result.fields_applied ~= nil, "fields_applied present when fields given")
  assertEq(result.fields_applied.applied.compensate_hilite_pres, false,
    "fields_applied echoes the applied field")
end

do
  local ok, err = pcall(internals.methods.dev_add_instance, {op = "exposure", fields = "not-a-table"})
  assertTrue(not ok, "dev_add_instance errors when fields is not a table")
end

do
  local ok, err = pcall(internals.methods.dev_add_instance, {})
  assertTrue(not ok, "dev_add_instance errors without op")
end

-- ---- methods.dev_list_masks / dev_list_all_masks / dev_attach_mask /
-- dev_detach_mask: mask group management (2026-07-27) -- attach/detach an
-- EXISTING shape to a module's blend group without copying it. -----------
do
  develop_calls = {}
  local result = internals.methods.dev_list_masks({op = "retouch", instance = 0})
  assertEq(#result, 1, "dev_list_masks forwards the C result")
  assertEq(result[1].operation, "union", "dev_list_masks includes operation")
  local call = develop_calls[1]
  assertEq(call.name, "list_masks", "correct C function called")
  assertEq(call.args[1], "retouch", "op forwarded")
  assertEq(call.args[2], 0, "instance forwarded")
end

do
  local ok, err = pcall(internals.methods.dev_list_masks, {})
  assertTrue(not ok, "dev_list_masks errors without op")
end

do
  develop_calls = {}
  local result = internals.methods.dev_list_all_masks({})
  assertEq(#result, 1, "dev_list_all_masks forwards the C result")
  assertEq(result[1].formid, 42, "dev_list_all_masks includes formid")
  assertEq(result[1].used_by[1].op, "retouch", "dev_list_all_masks includes used_by")
  assertEq(develop_calls[1].name, "list_all_masks", "correct C function called")
end

do
  develop_calls = {}
  local result = internals.methods.dev_attach_mask({op = "exposure", instance = 1, formid = 42})
  assertEq(result.ok, true, "dev_attach_mask reports ok")
  local call = develop_calls[1]
  assertEq(call.name, "attach_mask", "correct C function called")
  assertEq(call.args[1], "exposure", "op forwarded")
  assertEq(call.args[2], 1, "instance forwarded")
  assertEq(call.args[3], 42, "formid forwarded")
  assertTrue(call.args[4] == nil, "operation omitted forwards nil (C defaults to union)")
end

do
  develop_calls = {}
  internals.methods.dev_attach_mask({op = "exposure", instance = 1, formid = 42, operation = "difference"})
  local call = develop_calls[1]
  assertEq(call.args[4], "difference", "operation forwarded when given")
end

do
  local ok, err = pcall(internals.methods.dev_attach_mask, {op = "exposure", instance = 1})
  assertTrue(not ok, "dev_attach_mask errors without formid")
  assertTrue(string.find(err or "", "formid") ~= nil, "error mentions formid")
end

do
  local ok, err = pcall(internals.methods.dev_attach_mask, {formid = 42})
  assertTrue(not ok, "dev_attach_mask errors without op")
end

do
  develop_calls = {}
  local result = internals.methods.dev_detach_mask({op = "exposure", instance = 1, formid = 42})
  assertEq(result.ok, true, "dev_detach_mask reports ok")
  local call = develop_calls[1]
  assertEq(call.name, "detach_mask", "correct C function called")
  assertEq(call.args[3], 42, "formid forwarded")
end

do
  local ok, err = pcall(internals.methods.dev_detach_mask, {op = "exposure", instance = 1})
  assertTrue(not ok, "dev_detach_mask errors without formid")
end

-- ---- methods.dev_get_mask / dev_rename_mask (2026-07-31) -- mask geometry
-- read + rename, closing the gap a bugreport found: get_mask already
-- existed in C (registered) but was never wired through the Lua bridge. ----
do
  develop_calls = {}
  local result = internals.methods.dev_get_mask({mask_id = 42})
  assertEq(result.mask_id, 42, "dev_get_mask forwards the C result")
  assertEq(result.points[1].corner[1], 0.1, "dev_get_mask includes point geometry")
  local call = develop_calls[1]
  assertEq(call.name, "get_mask", "correct C function called")
  assertEq(call.args[1], 42, "mask_id forwarded")
end

do
  local ok, err = pcall(internals.methods.dev_get_mask, {})
  assertTrue(not ok, "dev_get_mask errors without mask_id")
end

do
  develop_calls = {}
  local result = internals.methods.dev_rename_mask({mask_id = 42, name = "model body"})
  assertEq(result.ok, true, "dev_rename_mask reports ok")
  assertEq(result.name, "model body", "dev_rename_mask returns the new name")
  local call = develop_calls[1]
  assertEq(call.name, "rename_mask", "correct C function called")
  assertEq(call.args[1], 42, "mask_id forwarded")
  assertEq(call.args[2], "model body", "name forwarded")
end

do
  local ok, err = pcall(internals.methods.dev_rename_mask, {mask_id = 42})
  assertTrue(not ok, "dev_rename_mask errors without name")
end

do
  local ok, err = pcall(internals.methods.dev_rename_mask, {name = "x"})
  assertTrue(not ok, "dev_rename_mask errors without mask_id")
end

do
  develop_calls = {}
  local result = internals.methods.dev_delete_mask({mask_id = 42})
  assertEq(result.ok, true, "dev_delete_mask reports ok")
  local call = develop_calls[1]
  assertEq(call.name, "delete_mask", "correct C function called")
  assertEq(call.args[1], 42, "mask_id forwarded")
end

do
  local ok, err = pcall(internals.methods.dev_delete_mask, {})
  assertTrue(not ok, "dev_delete_mask errors without mask_id")
end

-- ---- methods.dev_current_image sidecar field (2026-07-31) -- bugreport:
-- export_images silently exported the base/version-0 duplicate's sidecar
-- instead of the one open in darkroom. dev_current_image now merges in
-- image.sidecar (stock darktable Lua field) so a caller can pass the exact
-- sidecar to export_images's xmp_paths. -------------------------------------
do
  develop_calls = {}
  local result = internals.methods.dev_current_image({})
  assertEq(result.has_image, true, "dev_current_image forwards has_image")
  assertEq(result.id, 101, "dev_current_image forwards id")
  assertEq(result.sidecar, "/photos/DSC_0001.NEF.xmp", "dev_current_image merges in sidecar")
  assertEq(develop_calls[1].name, "current_image", "correct C function called")
end

do
  -- has_image=false: must NOT attempt a database lookup at all.
  local original_current_image = stub_dt.develop.current_image
  stub_dt.develop.current_image = function() return {has_image = false, id = -1} end
  local result = internals.methods.dev_current_image({})
  assertEq(result.has_image, false, "dev_current_image forwards has_image=false")
  assertTrue(result.sidecar == nil, "no sidecar merged in when no image is open")
  stub_dt.develop.current_image = original_current_image
end

-- ---- methods.tag_photo ------------------------------------------------------
do
  local result = internals.methods.tag_photo({photo_ids = {"101", "102"}, tags = {"keep"}})
  assertEq(result.updated, 2, "tag_photo updated count")
  assertEq(#result.tags_created, 1, "tag_photo created one new tag")
  assertEq(result.tags_created[1], "keep", "tag_photo reports created tag name")
  assertEq(#tags_by_name["keep"], 2, "tag 'keep' now attached to 2 images")
end

do
  -- Re-attaching an existing tag must not report it as newly created, and
  -- attaching to a photo that's already tagged must not duplicate the entry.
  local result = internals.methods.tag_photo({photo_ids = {"101", "999"}, tags = {"keep"}})
  assertEq(#result.tags_created, 0, "tag_photo does not re-create an existing tag")
  assertEq(result.updated, 1, "tag_photo only counts photos that exist")
  assertEq(result.missing_photos[1], "999", "tag_photo reports missing photo id")
  assertEq(#tags_by_name["keep"], 2, "re-attaching an existing tag does not duplicate")
end

do
  local result = internals.methods.tag_photo({photo_ids = {"101"}, remove_tags = {"keep"}})
  assertEq(result.updated, 1, "tag_photo remove_tags updated count")
  assertEq(#tags_by_name["keep"], 1, "detaching removes image from tag")
end

do
  local resp = internals.handle({id = "tp1", method = "tag_photo", params = {photo_ids = {}}})
  assertTrue(resp.error ~= nil, "tag_photo errors on empty photo_ids")
  assertTrue(string.find(resp.error or "", "photo_ids") ~= nil, "error mentions photo_ids")
end

do
  local resp = internals.handle({id = "tp2", method = "tag_photo", params = {photo_ids = {"101"}}})
  assertTrue(resp.error ~= nil, "tag_photo errors when neither tags nor remove_tags given")
end

-- ---- methods.list_collections -----------------------------------------------
do
  local result = internals.methods.list_collections({})
  assertEq(result.count, 1, "list_collections returns one tag so far")
  assertEq(result.collections[1].name, "keep", "list_collections returns tag name")
  assertEq(result.collections[1].count, 1, "list_collections returns tag photo count")
end

do
  local result = internals.methods.list_collections({filter = "nope"})
  assertEq(result.count, 0, "list_collections filter excludes non-matching tags")
end

-- ---- methods.list_photos_in_collection --------------------------------------
do
  local result = internals.methods.list_photos_in_collection({collection = "keep"})
  assertTrue(result.found, "list_photos_in_collection finds existing tag")
  assertEq(result.count, 1, "list_photos_in_collection returns one photo")
  assertEq(result.photos[1].id, "102", "list_photos_in_collection returns correct photo id")
  assertEq(result.photos[1].path, "/photos/DSC_0002.NEF",
    "list_photos_in_collection returns absolute file path")
end

do
  local result = internals.methods.list_photos_in_collection({collection = "nonexistent"})
  assertTrue(not result.found, "list_photos_in_collection reports not found for unknown tag")
  assertEq(result.count, 0, "list_photos_in_collection returns zero photos for unknown tag")
end

-- ---- methods.import_batch --------------------------------------------------
do
  -- Stub dt.database.import to record args and return a fake list.
  -- darktable's real API takes only a path string; recursion is governed by
  -- a darktable preference, not a per-call argument.
  local recorded = {}
  local stub_imported = {{}, {}, {}}  -- 3 fake images
  -- Save original (in case test_dispatcher runs other tests later that need it).
  local original_db = stub_dt.database
  stub_dt.database = setmetatable({
    import = function(path)
      table.insert(recorded, {path = path})
      return stub_imported
    end,
  }, {__index = original_db})

  local result = internals.methods.import_batch({source_path = "/tmp/foo", recursive = true})
  assertEq(result.imported, 3, "import_batch returns count of imported images")
  assertEq(result.source_path, "/tmp/foo", "import_batch returns source_path back")
  assertEq(result.recursive, true, "import_batch echoes recursive back")
  assertEq(#recorded, 1, "dt.database.import called once")
  assertEq(recorded[1].path, "/tmp/foo", "import called with source_path")

  -- Default recursive = true when not specified
  local r2 = internals.methods.import_batch({source_path = "/tmp/bar"})
  assertEq(r2.recursive, true, "import_batch defaults recursive=true")

  -- Single-image return (non-table) should count as 1.
  stub_dt.database = setmetatable({
    import = function(_) return {} end,  -- ensure a userdata-like (we use empty table)
  }, {__index = original_db})

  -- Restore.
  stub_dt.database = original_db
end

-- ---- import_batch error: missing source_path -------------------------------
do
  local resp = internals.handle({
    id = "ib1",
    method = "import_batch",
    params = {},  -- no source_path
  })
  assertEq(resp.id, "ib1", "import_batch error preserves id")
  assertTrue(resp.error ~= nil, "import_batch returns error when source_path missing")
  assertTrue(string.find(resp.error or "", "source_path") ~= nil,
    "error message mentions source_path")
end

-- ---- methods.list_styles ---------------------------------------------------
do
  -- Stub dt.styles with a tiny inventory.
  local fake_styles = {
    {name = "alpha", description = "first style"},
    {name = "beta", description = "second style"},
  }
  local original_styles = stub_dt.styles
  stub_dt.styles = fake_styles  -- behaves like a list under ipairs

  local result = internals.methods.list_styles({})
  assertEq(result.count, 2, "list_styles returns count of styles")
  assertEq(#result.styles, 2, "list_styles returns table of styles")
  assertEq(result.styles[1].name, "alpha", "first style name")
  assertEq(result.styles[1].description, "first style", "first style description")
  assertEq(result.styles[2].name, "beta", "second style name")

  stub_dt.styles = original_styles
end

-- ---- methods.apply_preset --------------------------------------------------
do
  -- Stub dt.styles + image:apply_style + dt.database lookup.
  local applied_to = {}
  local make_stub_image = function(id)
    local img = {id = id}
    function img:apply_style(s) table.insert(applied_to, {id = self.id, style = s.name}) end
    return img
  end

  local fake_styles = {
    {name = "alpha", description = "a"},
    {name = "beta", description = "b"},
  }
  local original_styles = stub_dt.styles
  local original_db = stub_dt.database
  stub_dt.styles = fake_styles

  local stub_db_inner = {}
  stub_db_inner[101] = make_stub_image(101)
  stub_db_inner[102] = make_stub_image(102)
  stub_dt.database = setmetatable({
    get_image = function(id) return stub_db_inner[id] end,
  }, {__index = stub_db_inner})

  local result = internals.methods.apply_preset({
    preset_name = "beta",
    photo_ids = {"101", "102"},
  })
  assertEq(result.applied, 2, "apply_preset applied count")
  assertEq(#result.missed, 0, "apply_preset no missed images")
  assertEq(result.preset_name, "beta", "apply_preset echoes preset_name")
  assertEq(#applied_to, 2, "apply_style invoked twice")
  assertEq(applied_to[1].style, "beta", "applied beta to first image")
  assertEq(applied_to[2].style, "beta", "applied beta to second image")

  stub_dt.styles = original_styles
  stub_dt.database = original_db
end

-- ---- apply_preset: missing image ID ----------------------------------------
do
  local fake_styles = {{name = "alpha", description = "a"}}
  local original_styles = stub_dt.styles
  local original_db = stub_dt.database
  stub_dt.styles = fake_styles
  stub_dt.database = {get_image = function(_) return nil end}

  local result = internals.methods.apply_preset({
    preset_name = "alpha",
    photo_ids = {"999"},
  })
  assertEq(result.applied, 0, "apply_preset applied=0 when image missing")
  assertEq(#result.missed, 1, "apply_preset reports missed image")
  assertEq(result.missed[1], "999", "missed list contains the photo_id")

  stub_dt.styles = original_styles
  stub_dt.database = original_db
end

-- ---- apply_preset: unknown style -------------------------------------------
do
  local fake_styles = {{name = "alpha", description = "a"}}
  local original_styles = stub_dt.styles
  stub_dt.styles = fake_styles

  local resp = internals.handle({
    id = "ap1",
    method = "apply_preset",
    params = {preset_name = "nonexistent", photo_ids = {"1"}},
  })
  assertEq(resp.id, "ap1", "preserves id on error")
  assertTrue(resp.error ~= nil, "returns error for unknown style")
  assertTrue(string.find(resp.error or "", "nonexistent") ~= nil, "names the missing style")

  stub_dt.styles = original_styles
end

-- ---- apply_preset: missing preset_name -------------------------------------
do
  local resp = internals.handle({
    id = "ap2",
    method = "apply_preset",
    params = {photo_ids = {"1"}},
  })
  assertTrue(resp.error ~= nil, "errors on missing preset_name")
  assertTrue(string.find(resp.error or "", "preset_name") ~= nil, "error mentions preset_name")
end

-- ---- apply_preset: empty photo_ids -----------------------------------------
do
  local resp = internals.handle({
    id = "ap3",
    method = "apply_preset",
    params = {preset_name = "alpha", photo_ids = {}},
  })
  assertTrue(resp.error ~= nil, "errors on empty photo_ids")
  assertTrue(string.find(resp.error or "", "photo_ids") ~= nil, "error mentions photo_ids")
end

-- ---- handle: known method --------------------------------------------------
do
  local resp = internals.handle({id = "abc", method = "view_photos", params = {limit = 1}})
  assertEq(resp.id, "abc", "handle preserves id")
  assertTrue(resp.result ~= nil, "handle known method returns result")
  assertTrue(resp.error == nil, "handle known method has no error")
end

-- ---- handle: unknown method ------------------------------------------------
do
  local resp = internals.handle({id = "xyz", method = "bogus", params = {}})
  assertEq(resp.id, "xyz", "handle preserves id on error")
  assertTrue(resp.error ~= nil, "handle unknown method returns error")
  assertTrue(string.find(resp.error, "bogus"), "error message names the method")
end

-- ---- scan_dir: full request/response round-trip ----------------------------
do
  local tmpdir = os.getenv("TMPDIR") or "/tmp"
  local test_dir = tmpdir .. "/darktable-mcp-lua-test-" .. tostring(os.time())
  os.execute("mkdir -p " .. test_dir)

  -- Reset stub state so view_photos in scan_dir sees the original data.
  images_by_id[101].rating = 5
  images_by_id[102].rating = 3

  -- Write a request file.
  local req_path = test_dir .. "/request-test001.json"
  local f = io.open(req_path, "w")
  f:write('{"id":"test001","method":"view_photos","params":{"limit":1}}')
  f:close()

  internals.scan_dir(test_dir)

  -- Verify request file was deleted.
  local req_check = io.open(req_path, "r")
  assertTrue(req_check == nil, "scan_dir deletes request file after processing")
  if req_check then req_check:close() end

  -- Verify response file appeared with correct content.
  local resp_path = test_dir .. "/response-test001.json"
  local resp_f = io.open(resp_path, "r")
  assertTrue(resp_f ~= nil, "scan_dir wrote response file")
  if resp_f then
    local content = resp_f:read("*a")
    resp_f:close()
    assertTrue(string.find(content, "test001"), "response contains request id")
    assertTrue(string.find(content, "result"), "response contains result field")
  end

  os.execute("rm -rf " .. test_dir)
end

-- ---- JSON round-trip with non-ASCII ----------------------------------------
do
  local original = {filter = "Тест", filename = "café.NEF"}
  local encoded = internals.json.encode(original)
  local decoded = internals.json.decode(encoded)
  assertEq(decoded.filter, "Тест", "non-ASCII Cyrillic round-trips")
  assertEq(decoded.filename, "café.NEF", "non-ASCII Latin-1 supplement round-trips")
end

-- ---- JSON \uXXXX escape decoding (matches what Python's json.dumps emits) -
do
  -- Python emits "Test" as ASCII, but emits "é" as é by default.
  local payload = '{"name":"caf\\u00e9.NEF"}'
  local decoded = internals.json.decode(payload)
  assertEq(decoded.name, "café.NEF", "\\u00e9 escape decodes to UTF-8")
end

-- ---- JSON control-byte escaping in encoder --------------------------------
do
  local with_ctrl = "ab\1cd"
  local encoded = internals.json.encode({s = with_ctrl})
  -- Encoder must escape \1 as  (otherwise Python's strict parser rejects).
  assertTrue(string.find(encoded, "\\u0001", 1, true) ~= nil,
    "encoder escapes 0x01 as \\u0001")
  local decoded = internals.json.decode(encoded)
  assertEq(decoded.s, with_ctrl, "control byte round-trips through escape")
end

-- ---- Report ----------------------------------------------------------------
if #failures > 0 then
  io.stderr:write("FAILED:\n")
  for _, msg in ipairs(failures) do
    io.stderr:write("  " .. msg .. "\n")
  end
  os.exit(1)
end
print("OK: all dispatcher tests passed")
os.exit(0)
