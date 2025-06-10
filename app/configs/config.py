import supervision as sv

# Core configuration
COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]
PRIMARY_COLOR = "#FF1493"
WHITE = "#FFFFFF"

# Reusable color objects
COLOR_PALETTE = sv.ColorPalette.from_hex(COLORS)
PRIMARY_SV_COLOR = sv.Color.from_hex(PRIMARY_COLOR)
WHITE_SV_COLOR = sv.Color.from_hex(WHITE)
VERTEX_COLORS = [sv.Color.from_hex(color) for color in COLORS]

# Common text settings
TEXT_CONFIG = {
    "text_color": WHITE_SV_COLOR,
    "text_thickness": 1,
    "text_padding": 5,
}
