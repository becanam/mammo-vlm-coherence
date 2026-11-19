"""
Convert mammography DICOMs to the 8-bit images the runner expects, following the conversion
described in the paper (Appendix B) and MammoVQA's released preprocessing. Starting from the source
DICOMs, each view is converted the same way for both RSNA and EMBED:

  1. apply the stored windowing (VOI-LUT) from the DICOM header;
  2. invert MONOCHROME1 images so soft tissue is bright on a dark background;
  3. rescale intensities to [0, 255] (8-bit grayscale);
  4. crop to the breast region (largest connected component of the foreground);
  5. downscale so the longer side is <= 1024 px;
  6. save as JPEG (quality 90), or PNG with --format png.

The grayscale->RGB channel replication is applied at inference time by the runner, not here, so a
single grayscale file is written per view. EMBED examinations are multi-view: each view is a
separate DICOM and is converted independently; the benchmark groups the views and the runner feeds
them jointly (see pipeline/run_eval.py).

Each output is named to match its MammoVQA benchmark path so the runner can find it. Pass --dataset
and the loop builds the right name automatically:

  RSNA   /Benchmark/RSNA/<pid>_<iid>_img/img.jpg  ->  <out>/<pid>_<iid>.png   (via --rsna_png)
  EMBED  /Benchmark/EMBED/<acc>/<name>.jpg        ->  <out>/<name>.jpg        (via --embed_png)

  # RSNA folder: builds <pid>_<iid>.png from the <pid>/<iid>.dcm layout
  python prepare_images.py --dataset rsna  --dicom_dir path/to/rsna_dicoms  --out_dir rsna_png

  # EMBED folder: the DICOM stem already equals the benchmark name
  python prepare_images.py --dataset embed --dicom_dir path/to/embed_dicoms --out_dir embed_png

  # one file (you name the output)
  python prepare_images.py --dicom path/to/img.dcm --out rsna_png/<pid>_<iid>.png
"""
import os
import argparse
import numpy as np
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from pathlib import Path

JPEG_QUALITY = 90
MAX_SIDE = 1024


def dicom_to_grayscale(dcm_path: str) -> np.ndarray:
    """Read a DICOM and return an 8-bit grayscale array: VOI-LUT applied, MONOCHROME1 inverted,
    rescaled to [0, 255]. No cropping or resizing yet."""
    dcm = pydicom.dcmread(dcm_path)
    arr = apply_voi_lut(dcm.pixel_array, dcm).astype(np.float32)
    if str(getattr(dcm, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr                      # invert: soft tissue bright on dark background
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)
    return (arr * 255.0).astype(np.uint8)


def breast_crop(arr: np.ndarray, trim: int = 10, thresh: int = 10) -> np.ndarray:
    """Crop to the breast region: trim a thin border, threshold the foreground, keep the largest
    connected component, and return the array cropped to its bounding box. Falls back to the input
    if no foreground is found."""
    h, w = arr.shape[:2]
    if h > 2 * trim and w > 2 * trim:
        inner = arr[trim:h - trim, trim:w - trim]
        off_y, off_x = trim, trim
    else:
        inner, off_y, off_x = arr, 0, 0
    mask = (inner > thresh).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return arr                                  # nothing above threshold -> leave as is
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh = (stats[largest, cv2.CC_STAT_LEFT], stats[largest, cv2.CC_STAT_TOP],
                    stats[largest, cv2.CC_STAT_WIDTH], stats[largest, cv2.CC_STAT_HEIGHT])
    return arr[off_y + y: off_y + y + bh, off_x + x: off_x + x + bw]


def downscale(arr: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = arr.shape[:2]
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        arr = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    return arr


def convert_one(dcm_path: str, out_path: str, max_side: int = MAX_SIDE, crop: bool = True):
    arr = dicom_to_grayscale(dcm_path)
    if crop:
        arr = breast_crop(arr)
    arr = downscale(arr, max_side)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.lower().endswith((".jpg", ".jpeg")):
        cv2.imwrite(out_path, arr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    else:
        cv2.imwrite(out_path, arr)


def out_stem(dcm: Path, dataset: str) -> str:
    """Output basename (no extension) that matches the MammoVQA path for this DICOM.
      rsna    -> '<patient>_<image>' from '<patient>/<image>.dcm' (benchmark uses <pid>_<iid>)
      embed   -> '<image>' (the DICOM stem already equals the benchmark name)
      generic -> '<image>' (the DICOM stem)"""
    if dataset == "rsna":
        return f"{dcm.parent.name}_{dcm.stem}"
    return dcm.stem


def main(a):
    if a.dicom:                                              # single file
        if not a.out:
            raise SystemExit("--out is required with --dicom")
        convert_one(a.dicom, a.out, a.max_side, crop=not a.no_crop)
        print(f"wrote {a.out}")
        return

    if not (a.dicom_dir and a.out_dir):
        raise SystemExit("pass --dicom (+ --out) for one file, or --dicom_dir + --out_dir for a folder")
    fmt = a.format or ("png" if a.dataset == "rsna" else "jpg")   # dataset default; --format overrides
    ext = fmt if fmt.startswith(".") else "." + fmt
    n = 0
    for dcm in sorted(Path(a.dicom_dir).rglob("*.dcm")):
        out_path = os.path.join(a.out_dir, out_stem(dcm, a.dataset) + ext)
        convert_one(str(dcm), out_path, a.max_side, crop=not a.no_crop)
        n += 1
    print(f"converted {n} DICOMs from {a.dicom_dir} -> {a.out_dir}/*{ext}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom", help="single DICOM file")
    ap.add_argument("--out", help="output path for --dicom (name it to match the benchmark path)")
    ap.add_argument("--dicom_dir", help="folder of DICOMs (searched recursively)")
    ap.add_argument("--out_dir", help="output folder for --dicom_dir")
    ap.add_argument("--dataset", choices=["generic", "rsna", "embed"], default="generic",
                    help="name/format outputs to match the MammoVQA paths: rsna -> <pid>_<iid>.png "
                         "(from <pid>/<iid>.dcm), embed -> <name>.jpg, generic -> <stem>")
    ap.add_argument("--format", default=None, help="png or jpg; overrides the per-dataset default")
    ap.add_argument("--max_side", type=int, default=MAX_SIDE, help="downscale so the longer side <= this")
    ap.add_argument("--no_crop", action="store_true", help="skip the breast-region crop")
    main(ap.parse_args())
