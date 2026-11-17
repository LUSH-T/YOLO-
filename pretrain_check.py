"""
训练前数据体检：确认新增数据被 YOLO 正确读取。

用法（在仓库根目录 D:\\DeepLearning\\ultralytics-8.4.202\\ultralytics-8.4.202 下）：
    python pretrain_check.py
    python pretrain_check.py --clean-cache     # 体检后顺手删掉 labels/*.cache

只依赖标准库，任何 Python 3 环境都能跑。
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

YAML = Path("ultralytics/cfg/datasets/robo.yaml")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_yaml(path: Path) -> dict:
    """极简解析：只取 path / train / val / names 几项，避免依赖 pyyaml。"""
    cfg = {"path": "", "train": "", "val": "", "names": {}}
    if not path.exists():
        sys.exit(f"[错误] 找不到数据集配置：{path}")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].rstrip()
        if not line.strip():
            continue
        m = re.match(r"^\s*(\d+)\s*:\s*(.+)$", line)          # names 下的 0: robot
        if m:
            cfg["names"][int(m.group(1))] = m.group(2).strip()
            continue
        m = re.match(r"^\s*(path|train|val)\s*:\s*(.+)$", line)
        if m:
            cfg[m.group(1)] = m.group(2).strip().strip("'\"")
    return cfg


def split_dir(root: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else root / p


def check_split(root: Path, name: str, img_dir: Path) -> dict:
    lbl_dir = Path(str(img_dir).replace("images", "labels", 1))
    imgs = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS) if img_dir.exists() else []
    clses = Counter()
    bad_lines = 0
    missing_lbl, orphan_lbl = [], []
    empty_lbl = 0

    for img in imgs:
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            missing_lbl.append(img.name)
            continue
        rows = [r for r in lbl.read_text(encoding="utf-8", errors="ignore").splitlines() if r.strip()]
        if not rows:
            empty_lbl += 1
            continue
        for r in rows:
            parts = r.split()
            try:
                cls = int(float(parts[0]))
                _ = [float(x) for x in parts[1:5]]
            except (ValueError, IndexError):
                bad_lines += 1
                continue
            clses[cls] += 1

    if lbl_dir.exists():
        for lbl in lbl_dir.rglob("*.txt"):
            if lbl.name == "classes.txt":  # 类别清单，不是标注
                continue
            if not (img_dir / (lbl.stem + ".jpg")).exists() and not any(
                (img_dir / (lbl.stem + e)).exists() for e in IMG_EXTS
            ):
                orphan_lbl.append(lbl.name)

    return {
        "split": name,
        "img_dir": img_dir,
        "n_img": len(imgs),
        "n_lbl": len(list(lbl_dir.rglob("*.txt"))) if lbl_dir.exists() else 0,
        "clses": clses,
        "missing": missing_lbl,
        "orphan": orphan_lbl,
        "empty": empty_lbl,
        "bad": bad_lines,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-cache", action="store_true", help="删除 labels/*.cache，强制 YOLO 重建索引")
    args = ap.parse_args()

    cfg = parse_yaml(YAML)
    root = Path(cfg["path"])
    names = cfg["names"]
    unknown_ids: set = set()

    print(f"数据集根目录: {root}")
    print(f"类别: {names}\n")

    for key in ("train", "val"):
        if not cfg[key]:
            continue
        r = check_split(root, key, split_dir(root, cfg[key]))
        print(f"[{r['split']}] {r['img_dir']}")
        print(f"  图片 {r['n_img']} 张 | 标注 {r['n_lbl']} 个")
        print(f"  实例数 " + (", ".join(f"{names.get(c, '?'+str(c))}={n}" for c, n in sorted(r["clses"].items())) or "0"))
        unknown_ids |= {c for c in r["clses"] if c not in names}
        if r["missing"]:
            print(f"  [警告] {len(r['missing'])} 张图没有标注（训练时会被静默跳过）: {r['missing'][:5]}")
        if r["orphan"]:
            print(f"  [提示] {len(r['orphan'])} 个标注找不到同名图片: {r['orphan'][:5]}")
        if r["empty"]:
            print(f"  [提示] {r['empty']} 个空标注文件（纯背景图，允许）")
        if r["bad"]:
            print(f"  [警告] {r['bad']} 行标注格式错误（不是 5/6 列的数值）")
        print()

    if unknown_ids:
        print(f"[错误] 标注里出现未定义的类别 id: {sorted(unknown_ids)}；请同步修改 robo.yaml 的 names")
    else:
        print("类别 id 校验通过")

    if args.clean_cache:
        removed = 0
        for c in root.rglob("*.cache"):
            c.unlink()
            removed += 1
        print(f"已删除 {removed} 个 cache 文件，下次训练会重建索引")


if __name__ == "__main__":
    main()
