"""Compose a wrist-camera inset for recorded robot replays."""

from PIL import Image, ImageDraw, ImageOps


def overlay_wrist_view(overview: Image.Image, wrist: Image.Image) -> None:
    width = round(overview.width * 0.30)
    height = round(wrist.height * width / wrist.width)
    inset = wrist.resize((width, height), Image.Resampling.LANCZOS)
    inset = ImageOps.expand(inset, border=2, fill="white")
    margin = max(4, overview.width // 100)
    left = overview.width - inset.width - margin
    top = overview.height - inset.height - margin
    overview.paste(inset, (left, top))
    draw = ImageDraw.Draw(overview)
    draw.rectangle((left + 2, top + 2, left + 45, top + 18), fill="#182531")
    draw.text((left + 7, top + 4), "Wrist", fill="white")
