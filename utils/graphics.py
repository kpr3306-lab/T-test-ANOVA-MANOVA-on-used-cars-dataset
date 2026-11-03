"""Inline SVG illustrations: side-profile cars per body type and the MotorIQ logo."""
import itertools

# Categorical order validated for the dark surface (#252B32) — keep it fixed per body type.
BODY_COLORS = {"Hatchback": "#E0682A", "Sedan": "#4A8CD0", "SUV": "#2FA878", "MUV": "#A774DB"}
BODY_BLURBS = {
    "Hatchback": "Compact, urban",
    "Sedan": "Long, elegant",
    "SUV": "Elevated, muscular",
    "MUV": "Spacious, family",
}

# body path, window path, pillar x positions, wheel x positions, wheel radius, ground y
_SHAPES = {
    "Hatchback": (
        "M26 64 L24 36 Q25 22 40 21 L104 20 Q114 20 122 27 L140 40 Q168 42 176 48 L178 60 Q178 64 174 64 Z",
        "M32 39 L35 27 Q36 25 40 25 L101 25 Q108 25 113 29 L124 39 Z",
        [74], [56, 150], 13, 64,
    ),
    "Sedan": (
        "M12 62 L12 50 Q14 44 28 42 L54 40 L72 25 Q77 21 88 21 L124 21 Q132 21 138 27 L152 40 Q178 42 186 48 L188 58 Q188 62 184 62 Z",
        "M60 39 L75 27 Q78 25 86 25 L121 25 Q127 25 131 29 L141 39 Z",
        [102], [46, 156], 13, 62,
    ),
    "SUV": (
        "M16 64 L15 34 Q16 21 30 20 L118 19 Q126 19 132 25 L146 38 Q176 40 184 46 L186 60 Q186 64 182 64 Z",
        "M24 36 L25 27 Q26 25 30 25 L114 25 Q120 25 125 30 L131 36 Z",
        [66, 100], [48, 154], 15, 64,
    ),
    "MUV": (
        "M14 64 L13 32 Q14 20 28 19 L122 18 Q130 18 138 24 L160 38 Q180 42 186 48 L188 60 Q188 64 184 64 Z",
        "M22 35 L23 26 Q24 24 28 24 L120 23 Q127 23 132 28 L142 35 Z",
        [54, 90, 122], [46, 158], 14, 64,
    ),
}

_ids = itertools.count()


def car_svg(body, color=None, width=200):
    """Side-profile car with metallic gradient, glass, rims, lights and ground shadow."""
    body = body if body in _SHAPES else "Hatchback"
    color = color or BODY_COLORS[body]
    path, glass, pillars, wheels, r, gy = _SHAPES[body]
    uid = next(_ids)  # gradient ids must be unique when many SVGs share one page
    front = max(wheels) + 24
    rear = min(wheels) - 26
    rails = '<path d="M34 17 L112 16" stroke="#AAB4C0" stroke-width="2" stroke-linecap="round"/>' if body == "SUV" else ""
    pillar_lines = "".join(
        f'<line x1="{x}" y1="{gy - 44}" x2="{x}" y2="{gy - 25}" stroke="{color}" stroke-width="3"/>' for x in pillars
    )
    wheel_svg = "".join(
        f'<circle cx="{x}" cy="{gy}" r="{r + 3}" fill="#171B20"/>'
        f'<circle cx="{x}" cy="{gy}" r="{r}" fill="#0E1114"/>'
        f'<circle cx="{x}" cy="{gy}" r="{r * 0.55:.1f}" fill="url(#rim{uid})"/>'
        f'<circle cx="{x}" cy="{gy}" r="{r * 0.18:.1f}" fill="#2A3038"/>'
        for x in wheels
    )
    return (
        f'<svg class="car-svg" viewBox="0 0 200 84" width="{width}" role="img" aria-label="{body} illustration" xmlns="http://www.w3.org/2000/svg">'
        f'<defs>'
        f'<linearGradient id="shine{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="#fff" stop-opacity=".45"/><stop offset=".45" stop-color="#fff" stop-opacity=".05"/>'
        f'<stop offset=".55" stop-color="#000" stop-opacity=".05"/><stop offset="1" stop-color="#000" stop-opacity=".4"/></linearGradient>'
        f'<linearGradient id="glass{uid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="#9FB4C8"/><stop offset="1" stop-color="#2B3540"/></linearGradient>'
        f'<radialGradient id="rim{uid}"><stop offset="0" stop-color="#E4E8EC"/><stop offset="1" stop-color="#6B7682"/></radialGradient>'
        f'<radialGradient id="shadow{uid}"><stop offset="0" stop-color="#000" stop-opacity=".55"/><stop offset="1" stop-color="#000" stop-opacity="0"/></radialGradient>'
        f'</defs>'
        f'<ellipse cx="100" cy="{gy + r - 1}" rx="92" ry="6" fill="url(#shadow{uid})"/>'
        f'{rails}'
        f'<path d="{path}" fill="{color}"/>'
        f'<path d="{path}" fill="url(#shine{uid})"/>'
        f'<path d="{glass}" fill="url(#glass{uid})"/>'
        f'{pillar_lines}'
        f'<path d="M{rear + 8} {gy - 20} L{front - 10} {gy - 20}" stroke="#000" stroke-opacity=".2" stroke-width="1.2"/>'
        f'<rect x="{front - 4}" y="{gy - 17}" width="9" height="4" rx="2" fill="#FFE7B0"/>'
        f'<rect x="{rear - 2}" y="{gy - 26}" width="5" height="5" rx="1.5" fill="#FF4D4D"/>'
        f'{wheel_svg}'
        f'</svg>'
    )


def logo_svg(size=40):
    """Speedometer dial wrapped around a tiny car silhouette."""
    return (
        f'<svg viewBox="0 0 48 48" width="{size}" height="{size}" role="img" aria-label="MotorIQ logo" xmlns="http://www.w3.org/2000/svg">'
        '<defs><linearGradient id="logo-g" x1="0" y1="0" x2="1" y2="1">'
        '<stop offset="0" stop-color="#F59A5F"/><stop offset="1" stop-color="#D3561B"/></linearGradient></defs>'
        '<rect width="48" height="48" rx="12" fill="#252B32"/>'
        '<path d="M9 31 A15 15 0 1 1 39 31" fill="none" stroke="#3A424C" stroke-width="4" stroke-linecap="round"/>'
        '<path d="M9 31 A15 15 0 0 1 33.5 14.5" fill="none" stroke="url(#logo-g)" stroke-width="4" stroke-linecap="round"/>'
        '<line x1="24" y1="29" x2="32" y2="17" stroke="#F4F5F6" stroke-width="2.4" stroke-linecap="round"/>'
        '<circle cx="24" cy="29" r="2.6" fill="#F4F5F6"/>'
        '<path d="M13 39 L15 35.5 Q16 34.5 18 34.5 L29 34.5 Q31 34.5 32.5 36 L35 37.5 Q36 38 36 39 Z" fill="url(#logo-g)"/>'
        '<circle cx="18" cy="39.5" r="2" fill="#F4F5F6"/><circle cx="31" cy="39.5" r="2" fill="#F4F5F6"/>'
        '</svg>'
    )
