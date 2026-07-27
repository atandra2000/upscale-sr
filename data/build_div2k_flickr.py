"""Download DIV2K + Flickr2K + DF2K HR images → webdataset shards or flat dir.

Idempotent + resumable.  Writes ``<out>/HR-######.tar`` shards (webdataset
backend) or ``<out>/hr/`` flat PNGs (flat backend), plus a ``manifest.json``
with the HR count.
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

from PIL import Image

SOURCES = {
    # DIV2K train + val (HTQ high-quality split, ~2K photos)
    "div2k_train": {
        "url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "subdir": "DIV2K_train_HR",
    },
    "div2k_val": {
        "url": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
        "subdir": "DIV2K_valid_HR",
    },
    # Flickr2K — distributed as a zip of HR JPEGs (legacy CVL mirror)
    "flickr2k": {
        "url": "http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar",
        "subdir": "Flickr2K_HR",
    },
}

VAL_SPLIT = 100        # DIV2K-valid: 100 images → fixed val set (DESIGN §3)
EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


def _download(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"  cached: {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url} → {dst}")
    urllib.request.urlretrieve(url, dst)


def _extract(archive: Path, out_dir: Path) -> Path:
    """Extract a .zip/.tar archive to ``out_dir`` (idempotent)."""
    marker = out_dir / ".extracted"
    if marker.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive} → {out_dir}")
    if archive.suffix == ".zip":
        import zipfile
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    else:
        with tarfile.open(archive) as t:
            t.extractall(out_dir)
    marker.touch()
    return out_dir


def _iter_hr_images(raw_root: Path):
    """Yield (name, PIL.Image) for every HR image under ``raw_root``."""
    for p in sorted(raw_root.rglob("*")):
        if p.suffix.lower() in EXTS:
            try:
                yield p.stem, Image.open(p).convert("RGB")
            except Exception as e:
                print(f"  skip {p}: {e}")


def _write_flat(val_imgs, train_imgs, out: Path):
    hr_dir = out / "hr"
    hr_dir.mkdir(parents=True, exist_ok=True)
    val_dir = out / "div2k_val"
    val_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for name, img in train_imgs:
        img.save(hr_dir / f"{name}.png")
        n += 1
    for name, img in val_imgs:
        img.save(val_dir / f"{name}.png")
    return n


def _write_wds_shards(val_imgs, train_imgs, out: Path, shard_size: int):
    shard_dir = out / "hr_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    val_dir = out / "div2k_val"
    val_dir.mkdir(parents=True, exist_ok=True)
    n, shard_idx, buf = 0, 0, []
    def flush(buf, idx):
        if not buf:
            return 0
        path = shard_dir / f"HR-{idx:06d}.tar"
        with tarfile.open(path, "w") as t:
            for name, img in buf:
                b = io.BytesIO()
                img.save(b, format="PNG")
                info = tarfile.TarInfo(f"{name}.png")
                info.size = b.tell()
                b.seek(0)
                t.addfile(info, b)
        return len(buf)
    for name, img in train_imgs:
        buf.append((name, img))
        n += 1
        if len(buf) >= shard_size:
            flush(buf, shard_idx); shard_idx += 1; buf = []
    flush(buf, shard_idx)
    for name, img in val_imgs:
        img.save(val_dir / f"{name}.png")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--backend", choices=["webdataset", "flat"], default="webdataset")
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--raw-cache", default=None,
                    help="where to cache the zips/tars (default: <out>/_raw)")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    raw_root = Path(args.raw_cache) if args.raw_cache else (out / "_raw")

    train_imgs, val_imgs = [], []
    for key, spec in SOURCES.items():
        archive = raw_root / Path(spec["url"]).name
        try:
            _download(spec["url"], archive)
            extracted = _extract(archive, raw_root / spec["subdir"])
        except Exception as e:
            print(f"  [WARN] could not fetch {key} ({e}) — skipping.")
            continue
        imgs = list(_iter_hr_images(extracted))
        print(f"  {key}: {len(imgs)} HR images")
        if key == "div2k_val":
            val_imgs.extend(imgs[:VAL_SPLIT])
        else:
            train_imgs.extend(imgs)

    if not train_imgs:
        raise RuntimeError("No training HR images downloaded — check network/URLs.")

    print(f"Writing {len(train_imgs)} train + {len(val_imgs)} val via {args.backend}…")
    if args.backend == "webdataset":
        n = _write_wds_shards(val_imgs, train_imgs, out, args.shard_size)
    else:
        n = _write_flat(val_imgs, train_imgs, out)

    manifest = {"n_train": n, "n_val": len(val_imgs), "backend": args.backend,
                "shard_size": args.shard_size}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. n_train={n} n_val={len(val_imgs)} → {out}/manifest.json")
    assert n + len(val_imgs) > 0
    # DESIGN §3 acceptance: > 4000 HR total when all three sources download
    print(f"Total HR images: {n + len(val_imgs)} (target ≥ 4000 when all sources fetch)")


if __name__ == "__main__":
    main()