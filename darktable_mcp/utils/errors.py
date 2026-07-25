"""Custom exceptions for darktable MCP server."""


class DarktableMCPError(Exception):
    """Base exception for darktable MCP server."""

    pass


class DarktableNotFoundError(DarktableMCPError):
    """Raised when darktable executable is not found."""

    pass


class DarktableLuaError(DarktableMCPError):
    """Raised when darktable Lua script execution fails."""

    pass


class InvalidRatingError(DarktableMCPError):
    """Raised when an invalid rating is provided."""

    pass


class PhotoNotFoundError(DarktableMCPError):
    """Raised when a specified photo is not found."""

    pass


class ValidationError(DarktableMCPError):
    """Raised when input validation fails."""

    pass


class ExportError(DarktableMCPError):
    """Raised when photo export fails."""

    pass


class SegmentationServiceError(DarktableMCPError):
    """Raised when the SAM2 segmentation sidecar (T2.1) can't produce a
    polygon: its venv is missing, the subprocess crashed/timed out, or it
    returned something that isn't the expected JSON contract. Callers
    (mask_object, T2.3) must treat this as "no polygon available" and must
    NOT have created any darkroom state yet when it's raised -- segmentation
    runs before add_instance in the orchestration."""

    pass


class MattingServiceError(DarktableMCPError):
    """Raised when the MODNet matting sidecar (T3.2) can't produce an alpha
    matte: its venv is missing, the ONNX checkpoint is absent, the
    subprocess crashed/timed out, or it returned something that isn't the
    expected JSON contract. Unlike SegmentationServiceError, there is
    deliberately NO lightweight in-process fallback here -- a hard-edged
    GrabCut/threshold stand-in is not a matte (it cannot produce the soft,
    continuous alpha this whole path exists for), so any failure just means
    "install/point at the sidecar" rather than "degrade to a rougher tier".
    Callers (mask_raster, T3.3) must treat this as "no matte available" and
    must NOT have created any darkroom state yet when it's raised --
    matting runs before add_instance(rasterfile) in the orchestration."""

    pass
