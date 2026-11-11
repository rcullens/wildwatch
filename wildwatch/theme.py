"""Shared visual theme, copied from Pocket LLM / Pixel Live Translate.

Source of truth (identical in both apps):

  rcullens/pocket-llm-ui  styles.css  :root
    /* Ported from ProxyScrapeStudio: cyan + purple dark theme */
    --bg:#14122B; --panel:#1C1A3A; --panel2:#221F47; --line:#332E66;
    --txt:#E8E6F5; --mut:#9C97C9; --accent:#22D3EE; --purple:#7C4DFF;
    --user:#2A2656; --bot:#1C1A3A; --danger:#F472B6;
    --grad: linear-gradient(90deg,#22D3EE 0%,#7C4DFF 100%);
    --grad-hover: linear-gradient(90deg,#4FE0F3 0%,#9470FF 100%);
    send-on text on the gradient: #0B0A1A
    status.ok: #34D399

  rcullens/pixel-live-translate
    app/src/main/res/values/colors.xml
    app/src/main/java/com/pixellivetranslate/app/ui/Theme.kt
    (comment: "Tokens copied from pocket-llm-ui styles.css")

Dark is the product default. Category colors on the HUD use only tokens
that already exist in this palette — nothing was invented to "look close."
"""

from __future__ import annotations


def _bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


# Dark theme tokens (product default).
BG = "#14122B"
PANEL = "#1C1A3A"
PANEL2 = "#221F47"
LINE = "#332E66"
TXT = "#E8E6F5"
MUT = "#9C97C9"
ACCENT = "#22D3EE"
PURPLE = "#7C4DFF"
USER = "#2A2656"
BOT = "#1C1A3A"
DANGER = "#F472B6"
SEND_ON = "#0B0A1A"
STATUS_OK = "#34D399"
ACCENT_HOVER = "#4FE0F3"
PURPLE_HOVER = "#9470FF"
GRAD = "linear-gradient(90deg, #22D3EE 0%, #7C4DFF 100%)"
GRAD_HOVER = "linear-gradient(90deg, #4FE0F3 0%, #9470FF 100%)"

# OpenCV is BGR.
BGR_BG = _bgr(BG)
BGR_PANEL = _bgr(PANEL)
BGR_PANEL2 = _bgr(PANEL2)
BGR_LINE = _bgr(LINE)
BGR_TXT = _bgr(TXT)
BGR_MUT = _bgr(MUT)
BGR_ACCENT = _bgr(ACCENT)
BGR_PURPLE = _bgr(PURPLE)
BGR_DANGER = _bgr(DANGER)
BGR_SEND_ON = _bgr(SEND_ON)
BGR_STATUS_OK = _bgr(STATUS_OK)
BGR_ACCENT_HOVER = _bgr(ACCENT_HOVER)

# Semantic HUD categories — mapped onto existing palette tokens only.
CATEGORY_COLOR = {
    "predator": BGR_DANGER,       # #F472B6
    "game": BGR_PURPLE,           # #7C4DFF
    "person": BGR_ACCENT,         # #22D3EE
    "vehicle": BGR_ACCENT_HOVER,  # #4FE0F3  (hover cyan, still in-palette)
    "animal": BGR_STATUS_OK,      # #34D399
    "other": BGR_MUT,             # #9C97C9
}

CSS_VARS = f"""
    --bg:{BG}; --panel:{PANEL}; --panel2:{PANEL2}; --line:{LINE};
    --txt:{TXT}; --mut:{MUT}; --accent:{ACCENT}; --purple:{PURPLE};
    --user:{USER}; --bot:{BOT}; --danger:{DANGER};
    --send-on:{SEND_ON}; --ok:{STATUS_OK};
    --grad:{GRAD}; --grad-hover:{GRAD_HOVER};
"""
