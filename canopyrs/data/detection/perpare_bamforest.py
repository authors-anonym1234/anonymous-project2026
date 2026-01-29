import argparse
import json
import os
from pathlib import Path

from geodataset.utils import CocoNameConvention, TileNameConvention


def _parse_bam_basename_to_col_row(basename: str) -> tuple[int, int]:
    """Parse BAMFORESTS tile basename like 'Stadtwald_170_0.tif' -> (170, 0).

    Uses rsplit to tolerate site names with underscores.
    """
    stem = Path(basename).stem
    try:
        _, col_str, row_str = stem.rsplit("_", 2)
        return int(col_str), int(row_str)
    except Exception as e:
        raise ValueError(f"Can't parse col/row from '{basename}'") from e


def _safe_link(src: Path, dst: Path, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        if not force:
            return
        dst.unlink()

    os.symlink(src, dst)


def _rewrite_coco_json(
    src_json: Path,
    dst_json: Path,
    filename_mapping: dict[str, str],
) -> None:
    with src_json.open("r") as f:
        coco = json.load(f)

    missing = 0
    for img in coco.get("images", []):
        old = img.get("file_name")
        if old in filename_mapping:
            img["file_name"] = filename_mapping[old]
        else:
            missing += 1

    if missing:
        raise RuntimeError(
            f"{src_json.name}: {missing} images had no filename mapping (tiles missing?)"
        )

    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with dst_json.open("w") as f:
        json.dump(coco, f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert BAMFORESTS COCO2048 layout into CanopyRS 'preprocessed' layout "
            "(same structure as NeonTreeEvaluation: location/product/tiles/<fold> + *_coco_*.json)."
        )
    )
    p.add_argument(
        "--bam_root",
        type=Path,
        required=True,
        help="Path to BAMFORESTS coco2048 root (contains annotations/, train2023/, val2023/, test2023/)",
    )
    p.add_argument(
        "--raw_data_root",
        type=Path,
        default=None,
        help=(
            "Destination raw_data_root used by CanopyRS benchmarker. "
            "Defaults to bam_root.parent (e.g. /data/xxx/dataset)."
        ),
    )
    p.add_argument(
        "--ground_resolution",
        type=float,
        default=0.017,
        help="Ground resolution in meters/pixel used in naming conventions (default: 0.017).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing links/COCO files if present.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bam_root: Path = args.bam_root
    raw_data_root: Path = args.raw_data_root or bam_root.parent

    location = "bamforests"

    splits = [
        ("train", "BAMForest_train2023", bam_root / "train2023", bam_root / "annotations" / "instances_tree_train2023.json"),
        ("valid", "BAMForest_val2023", bam_root / "val2023", bam_root / "annotations" / "instances_tree_eval2023.json"),
        ("test", "BAMForest_TestSet1_2023", bam_root / "test2023" / "Test-Set-1", bam_root / "annotations" / "instances_tree_TestSet12023.json"),
        ("test", "BAMForest_TestSet2_2023", bam_root / "test2023" / "Test-Set-2", bam_root / "annotations" / "instances_tree_TestSet22023.json"),
    ]

    for fold, product, src_tiles, src_coco in splits:
        if not src_tiles.exists():
            raise FileNotFoundError(f"Missing tiles dir: {src_tiles}")
        if not src_coco.exists():
            raise FileNotFoundError(f"Missing COCO json: {src_coco}")

        prod_root = raw_data_root / location / product
        dst_tiles = prod_root / "tiles" / fold

        mapping: dict[str, str] = {}
        for src_tif in sorted(src_tiles.glob("*.tif")):
            col, row = _parse_bam_basename_to_col_row(src_tif.name)
            new_name = TileNameConvention.create_name(
                product_name=product,
                col=col,
                row=row,
                ground_resolution=args.ground_resolution,
                aoi=None,
            )
            mapping[src_tif.name] = new_name
            _safe_link(src_tif, dst_tiles / new_name, force=args.force)

        dst_coco_name = CocoNameConvention.create_name(
            product_name=product,
            fold=fold,
            ground_resolution=args.ground_resolution,
        )
        dst_coco = prod_root / dst_coco_name
        if dst_coco.exists() and not args.force:
            continue
        _rewrite_coco_json(src_coco, dst_coco, mapping)

    print("Done. BAMForest is now in NeonTreeEvaluation-style layout.")
    print(f"raw_data_root: {raw_data_root}")
    print(f"location: {location}")


if __name__ == "__main__":
    main()
