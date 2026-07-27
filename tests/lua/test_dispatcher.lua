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
  [101] = {id = 101, filename = "DSC_0001.NEF", path = "/photos", rating = 5},
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
  local ok, err = pcall(internals.methods.dev_retouch_add_shape, {
    op = "retouch", algorithm = "heal", shape_type = "path",
    target = {x = 0.4, y = 0.3}, source = {x = 0.35, y = 0.3}, radius = 0.02,
  })
  assertTrue(not ok, "dev_retouch_add_shape errors on unsupported shape_type")
  assertTrue(string.find(err or "", "circle") ~= nil, "error mentions circle")
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
    op = "retouch", formid = 42, target = {x = 0.4, y = 0.3}, radius = 0.02,
  })
  assertTrue(not ok, "dev_retouch_update_shape errors without a source point")
  assertTrue(string.find(err or "", "source") ~= nil, "error mentions source")
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
