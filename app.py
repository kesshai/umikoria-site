#!/usr/bin/env python3
"""
app.py

Web-приложение на Flask для пикселизации изображений:
- AJAX-превью «orig / mid / after»
- Dither, шумоподавление, posterize, outline
- Слияние похожих цветов
- Валидация параметров и защита от ошибок
"""
import os
from io import BytesIO
from flask import Flask, render_template, request, jsonify
import base64
import json
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import cv2
import scipy
from itertools import product

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

PALETTE_FILE = os.path.join(app.static_folder, 'palettes.json')
try:
    with open(PALETTE_FILE, 'r') as f:
        PALETTES = json.load(f)
except Exception:
    PALETTES = []


def merge_similar_colors(img: Image.Image, delta: int) -> Image.Image:

    if delta <= 0:
        return img
    arr = np.array(img)
    flat = arr.reshape(-1, arr.shape[2])
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    reps = []
    clusters = []
    for idx, col in enumerate(unique):
        placed = False
        for i, rep in enumerate(reps):
            if np.linalg.norm(col[:3] - rep[:3]) <= delta:
                clusters[i].append(idx)
                placed = True
                break
        if not placed:
            reps.append(col)
            clusters.append([idx])
    mapping = {}
    for rep, cluster in zip(reps, clusters):
        for cidx in cluster:
            mapping[cidx] = rep
    new_flat = np.array([mapping[i] for i in inverse], dtype=np.uint8)
    new_arr = new_flat.reshape(arr.shape)
    return Image.fromarray(new_arr, img.mode)


def detect_edges(img: Image.Image, thresh: int, dilate: int) -> Image.Image:

    arr = np.array(img)
    if arr.shape[2] == 4:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    lower = max(0, thresh // 2)
    upper = max(lower + 1, thresh)
    edges = cv2.Canny(gray, lower, upper)
    if dilate > 1:
        kernel = np.ones((dilate, dilate), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    return Image.fromarray(edges).convert('1')


def kCentroid(image: Image.Image, width: int, height: int, centroids: int):
    image = image.convert("RGB")
    downscaled = np.zeros((height, width, 3), dtype=np.uint8)
    wFactor = image.width / width
    hFactor = image.height / height
    for x, y in product(range(width), range(height)):
        tile = image.crop(
            (x * wFactor, y * hFactor, (x * wFactor) + wFactor, (y * hFactor) + hFactor)
        )
        tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert("RGB")
        color_counts = tile.getcolors()
        most_common_color = max(color_counts, key=lambda c: c[0])[1]
        downscaled[y, x, :] = most_common_color
    return Image.fromarray(downscaled, mode="RGB")


def pixel_detect(image: Image.Image):
    npim = np.array(image)[..., :3]
    hdiff = np.sqrt(np.sum((npim[:, :-1, :] - npim[:, 1:, :]) ** 2, axis=2))
    hsum = np.sum(hdiff, 0)
    vdiff = np.sqrt(np.sum((npim[:-1, :, :] - npim[1:, :, :]) ** 2, axis=2))
    vsum = np.sum(vdiff, 1)
    hpeaks, _ = scipy.signal.find_peaks(hsum, distance=1, height=0.0)
    vpeaks, _ = scipy.signal.find_peaks(vsum, distance=1, height=0.0)

    hspacing = np.diff(hpeaks)
    vspacing = np.diff(vpeaks)

    hspace = np.median(hspacing) if hspacing.size > 0 else image.width
    vspace = np.median(vspacing) if vspacing.size > 0 else image.height

    if not np.isfinite(hspace) or hspace <= 0:
        hspace = image.width
    if not np.isfinite(vspace) or vspace <= 0:
        vspace = image.height

    width = max(1, int(round(image.width / hspace)))
    height = max(1, int(round(image.height / vspace)))

    return kCentroid(image, width, height, 2)


def determine_best_k(image: Image.Image, max_k: int):

    image = image.convert("RGB")

    w, h = image.size
    max_side = max(w, h)
    if max_side > 64:
        scale = max_side / 64.0
        new_size = (max(1, int(w / scale)), max(1, int(h / scale)))
        small = image.resize(new_size, Image.BILINEAR)
    else:
        small = image

    pixels = np.array(small)
    pixel_indices = pixels.reshape(-1, 3)

    distortions = []
    prev_distortion = float("inf")
    for k in range(1, max_k + 1):
        quantized = small.quantize(colors=k, method=0, kmeans=k, dither=0)
        centroids = np.array(quantized.getpalette()[: k * 3]).reshape(-1, 3)
        distances = np.linalg.norm(pixel_indices[:, None] - centroids, axis=2)
        min_distances = np.min(distances, axis=1)
        distortion = float(np.sum(min_distances ** 2))
        distortions.append(distortion)

        if prev_distortion != float("inf"):
            if prev_distortion == 0:
                break
            improvement = (prev_distortion - distortion) / prev_distortion
            if improvement < 0.01:
                break
        prev_distortion = distortion

    distortions = np.array(distortions, dtype=float)
    if len(distortions) < 2:
        return min(len(distortions), max_k)

    rate_of_change = np.divide(
        np.diff(distortions),
        distortions[:-1],
        out=np.zeros_like(distortions[:-1]),
        where=distortions[:-1] != 0,
    )

    elbow_index = int(np.argmax(rate_of_change)) + 1
    return min(elbow_index + 1, max_k)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and request.headers.get('X-Requested-With'):
        try:
            file = request.files['image']
            img  = Image.open(file.stream).convert('RGBA')

            scale          = max(1, int(request.form.get('scale', 8)))
            colors         = min(max(2, int(request.form.get('colors', 16))), 256)
            use_dither     = bool(request.form.get('dither'))
            thresh         = min(max(0, int(request.form.get('outline_threshold', 40))), 255)
            dilate         = min(max(1, int(request.form.get('outline_dilate', 3))), 15)
            noise_radius   = min(max(0.0, float(request.form.get('noise_radius', 0))), 15.0)
            posterize_bits = min(max(1, int(request.form.get('posterize_bits', 8))), 8)
            brightness     = min(max(0.0, float(request.form.get('brightness', 1.0))), 5.0)
            saturation     = min(max(0.0, float(request.form.get('saturation', 1.0))), 5.0)
            contrast       = min(max(0.0, float(request.form.get('contrast', 1.0))), 5.0)
            merge_delta    = min(max(0, int(request.form.get('merge_delta', 0))), 100)
            auto_pixel     = bool(request.form.get('auto_pixel'))
            auto_palette   = bool(request.form.get('auto_palette'))
            auto_palette_max = min(max(2, int(request.form.get('auto_palette_max', 128))), 256)
            palette_raw    = request.form.get('palette')
            palette_idx    = int(palette_raw) if palette_raw not in (None, '', 'null') else None
            palette_colors = None
            if palette_idx is not None and 0 <= palette_idx < len(PALETTES):
                palette_colors = PALETTES[palette_idx].get('colors', [])

            buf_orig = BytesIO()
            img.save(buf_orig, 'PNG')
            orig_b64 = base64.b64encode(buf_orig.getvalue()).decode('utf-8')

            w, h  = img.size
            if auto_pixel:
                small = pixel_detect(img)
            else:
                small = img.resize((w//scale or 1, h//scale or 1), Image.NEAREST)

            if noise_radius > 0:
                small = small.filter(ImageFilter.GaussianBlur(radius=noise_radius))

            if posterize_bits < 8:
                rgb   = small.convert('RGB')
                post  = ImageOps.posterize(rgb, posterize_bits)
                alpha = small.split()[3]
                post  = post.convert('RGBA')
                post.putalpha(alpha)
                small = post

            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(small)
                small = enhancer.enhance(brightness)
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(small)
                small = enhancer.enhance(saturation)
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(small)
                small = enhancer.enhance(contrast)

            if auto_palette:
                colors = determine_best_k(small, auto_palette_max)

            buf_mid = BytesIO()
            small.save(buf_mid, 'PNG')
            mid_b64 = base64.b64encode(buf_mid.getvalue()).decode('utf-8')

            orig_rgb    = small.convert('RGB')
            dither_flag = Image.FLOYDSTEINBERG if use_dither else 0

            if palette_colors:
                pal_img = Image.new('P', (1, 1))
                flat = []
                for hx in palette_colors:
                    r = int(hx[1:3], 16)
                    g = int(hx[3:5], 16)
                    b = int(hx[5:7], 16)
                    flat.extend([r, g, b])
                pal_img.putpalette(flat + [0] * (768 - len(flat)))
                quant_base = orig_rgb.quantize(
                    palette=pal_img,
                    dither=dither_flag
                )
            else:
                quant_base  = orig_rgb.quantize(
                    colors=colors,
                    method=Image.MEDIANCUT,
                    dither=dither_flag
                )
            base_rgba   = quant_base.convert('RGBA')
            base_rgba.putalpha(small.split()[3])
            if palette_colors:
                quant_edges = orig_rgb.quantize(
                    palette=pal_img,
                    dither=dither_flag
                )
            else:
                quant_edges = orig_rgb.quantize(
                    palette=quant_base,
                    dither=dither_flag
                )
            edges_rgba = quant_edges.convert('RGBA')
            edges_rgba.putalpha(small.split()[3])

            mask = detect_edges(small, thresh, dilate)

            out = base_rgba.copy()
            out.paste(edges_rgba, mask=mask)
            out = merge_similar_colors(out, merge_delta)

            buf_after = BytesIO()
            out.save(buf_after, 'PNG')
            after_b64 = base64.b64encode(buf_after.getvalue()).decode('utf-8')

            return jsonify(
                orig = f"data:image/png;base64,{orig_b64}",
                mid  = f"data:image/png;base64,{mid_b64}",
                after= f"data:image/png;base64,{after_b64}"
            )

        except Exception as e:
            return jsonify(error=f"{type(e).__name__}: {e}")

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
