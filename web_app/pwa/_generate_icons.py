# -*- coding: utf-8 -*-
"""一次性脚本：从 web_app/logo.png（1024x1024）生成 PWA 各尺寸图标。
运行：python pwa/_generate_icons.py（在工作目录 web_app 下）
"""

import os

from PIL import Image, ImageOps


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE_DIR, "logo.png")
OUT_DIR = os.path.join(BASE_DIR, "pwa", "icons")

BG = (255, 255, 255, 255)          # 纯白背景
SAFE_RATIO = 0.62                  # maskable 安全区（防安卓裁切）


def make_icon(size, maskable=False):
    """生成一张指定尺寸的方形图标，白底 + logo 居中。"""
    logo = Image.open(SRC).convert("RGBA")
    canvas = Image.new("RGBA", (size, size), BG)
    if maskable:
        # 内容只占中央 ~62%，四周留白，避免被系统裁成圆形/圆角时切到 logo
        content = int(size * SAFE_RATIO)
    else:
        content = size
    logo = ImageOps.contain(logo, (content, content), method=Image.LANCZOS)
    canvas.paste(logo, ((size - logo.width) // 2, (size - logo.height) // 2), logo)
    return canvas.convert("RGB")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = [
        ("icon-192.png", 192, False),
        ("icon-512.png", 512, False),
        ("icon-192-maskable.png", 192, True),
        ("icon-512-maskable.png", 512, True),
        ("apple-touch-icon-180.png", 180, False),
        ("favicon-32.png", 32, False),
        ("favicon-16.png", 16, False),
    ]
    for name, size, maskable in jobs:
        path = os.path.join(OUT_DIR, name)
        make_icon(size, maskable=maskable).save(path, "PNG")
        print("generated", os.path.relpath(path, BASE_DIR), size, "x", size)


if __name__ == "__main__":
    main()
