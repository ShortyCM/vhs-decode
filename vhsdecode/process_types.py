"""Pickle-safe immutable value types used by the VHS decoder."""

from collections import namedtuple


# These classes must live at module scope. Windows' multiprocessing ``spawn``
# method reconstructs them by importing their defining module.
Options = namedtuple(
    "Options",
    [
        "diff_demod_check_value", "tape_format", "disable_comb", "nldeemp",
        "subdeemp", "disable_right_hsync", "disable_dc_offset",
        "fallback_vsync", "field_order_confidence", "saved_levels", "y_comb",
        "write_chroma", "color_under", "chroma_deemphasis_filter",
        "skip_hsync_refine", "hsync_refine_use_threshold", "export_raw_tbc",
        "fm_audio_notch", "chroma_audio_notch", "chroma_offset", "cti_mix",
        "cti_width", "ire0_adjust", "gnrc_afe", "relaxed_line0",
        "detect_chroma_track_phase", "enable_color_killer",
        "disable_burst_hsync", "disable_phase_correction", "secam_carrier_servo",
    ],
)
SysparamsConst = namedtuple(
    "SysparamsConst", "hz_ire vsync_hz vsync_ire ire0 vsync_pulse_us"
)
