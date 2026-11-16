"""Image-level worker functions executed inside Spark `mapInPandas`.

Design rule: Spark moves *paths*, never pixel bytes. SH17 images are up to 8192x5462, so shipping
`binaryFile.content` through Arrow would multiply memory use; each worker opens files itself.
"""
from __future__ import annotations

import hashlib
import io
import os
from typing import Iterator

import numpy as np
import pandas as pd

EXIF_ORIENTATION = 0x0112


def local_path(p: str) -> str:
    return p[5:] if p.startswith("file:") else p


def _seed(image_id: str, salt: int) -> int:
    return int(hashlib.md5(f"{image_id}:{salt}".encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- metadata
META_SCHEMA = ("image_id string, path string, bytes long, width int, height int, eff_width int, "
               "eff_height int, exif_orientation int, img_format string, mode string, "
               "read_ok boolean, error string")
_META_COLS = [c.split()[0] for c in META_SCHEMA.split(", ")]


def image_meta(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """Header-only read (no full decode): size, EXIF orientation, format, integrity check."""
    from PIL import Image

    for pdf in batches:
        rows = []
        for image_id, path, nbytes in zip(pdf["image_id"], pdf["path"], pdf["bytes"]):
            p = local_path(path)
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    o = int(im.getexif().get(EXIF_ORIENTATION, 1) or 1)
                    fmt, mode = im.format, im.mode
                    im.verify()  # structural check of the file without decoding pixels
                ew, eh = (h, w) if o in (5, 6, 7, 8) else (w, h)  # rotated 90/270 -> swap
                rows.append((image_id, p, nbytes, w, h, ew, eh, o, fmt, mode, True, None))
            except Exception as e:  # corrupt / truncated / unsupported
                rows.append((image_id, p, nbytes, None, None, None, None, None, None, None, False, str(e)[:300]))
        out = pd.DataFrame(rows, columns=_META_COLS)
        for c in ("width", "height", "eff_width", "eff_height", "exif_orientation"):
            out[c] = out[c].astype("Int32")
        out["bytes"] = out["bytes"].astype("Int64")
        yield out


# --------------------------------------------------------------------------- perceptual hash
PHASH_SCHEMA = "image_id string, phash string"


def image_phash(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    import imagehash
    from PIL import Image, ImageOps

    for pdf in batches:
        rows = []
        for image_id, path in zip(pdf["image_id"], pdf["path"]):
            try:
                with Image.open(path) as im:
                    im.draft("L", (256, 256))  # JPEG DCT-domain downscale: ~10-50x faster on 8K images
                    im = ImageOps.exif_transpose(im).convert("L")
                    rows.append((image_id, str(imagehash.phash(im))))  # 16 hex chars = 64 bits
            except Exception:
                rows.append((image_id, None))
        yield pd.DataFrame(rows, columns=["image_id", "phash"])


# --------------------------------------------------------------------------- export (resize + labels)
EXPORT_SCHEMA = "image_id string, split string, out_width int, out_height int, status string"


def degrade_cctv(im, rng: np.random.Generator):
    """Approximate a fixed industrial CCTV camera: resolution loss, defocus, low light, sensor noise,
    partial desaturation and heavy JPEG compression. Geometry is unchanged, so labels stay valid."""
    from PIL import Image, ImageFilter

    w, h = im.size
    s = rng.uniform(0.3, 0.55)
    im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if rng.random() < 0.6:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 2.0)))
    a = np.asarray(im).astype(np.float32) / 255.0
    a = np.power(a, rng.uniform(1.3, 2.2)) * rng.uniform(0.55, 0.9)        # under-exposure
    gray = a.mean(axis=2, keepdims=True)
    a = gray + (a - gray) * rng.uniform(0.3, 0.9)                           # washed-out colour
    a = a + rng.normal(0.0, rng.uniform(0.01, 0.04), a.shape)              # sensor noise
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(rng.integers(15, 40)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_export_fn(yolo_root: str, long_side: int, quality: int, skip_existing: bool,
                   build_cctv: bool, seed: int):
    """Factory -> mapInPandas function. Input columns: image_id, path, split, label_txt."""

    def _fn(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        from PIL import Image, ImageOps

        for pdf in batches:
            rows = []
            for r in pdf.itertuples(index=False):
                dst = os.path.join(yolo_root, r.split, "images", f"{r.image_id}.jpg")
                dst_lbl = os.path.join(yolo_root, r.split, "labels", f"{r.image_id}.txt")
                try:
                    need_img = not (skip_existing and os.path.exists(dst))
                    need_cctv = build_cctv and r.split == "test"
                    ow = oh = None
                    if need_img or need_cctv:
                        with Image.open(r.path) as im:
                            im.draft("RGB", (long_side, long_side))
                            # Ultralytics/OpenCV read images EXIF-corrected, so bake the rotation in once
                            im = ImageOps.exif_transpose(im).convert("RGB")
                            im.thumbnail((long_side, long_side), Image.LANCZOS)  # only ever downsizes
                            ow, oh = im.size
                            if need_img:
                                im.save(dst, "JPEG", quality=quality, optimize=True)
                            if need_cctv:
                                rng = np.random.default_rng(_seed(r.image_id, seed))
                                degrade_cctv(im, rng).save(
                                    os.path.join(yolo_root, "test_cctv", "images", f"{r.image_id}.jpg"),
                                    "JPEG", quality=95)
                    # normalized YOLO coords are invariant to uniform resizing -> labels copied as-is
                    txt = r.label_txt or ""  # empty file = background image (valid for Ultralytics)
                    with open(dst_lbl, "w") as f:
                        f.write(txt + ("\n" if txt else ""))
                    if build_cctv and r.split == "test":
                        with open(os.path.join(yolo_root, "test_cctv", "labels", f"{r.image_id}.txt"), "w") as f:
                            f.write(txt + ("\n" if txt else ""))
                    rows.append((r.image_id, r.split, ow, oh, "ok"))
                except Exception as e:
                    rows.append((r.image_id, r.split, None, None, f"error: {str(e)[:200]}"))
            out = pd.DataFrame(rows, columns=["image_id", "split", "out_width", "out_height", "status"])
            out["out_width"] = out["out_width"].astype("Int32")
            out["out_height"] = out["out_height"].astype("Int32")
            yield out

    return _fn
