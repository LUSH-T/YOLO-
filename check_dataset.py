"""数据集结构自检（不训练）。运行: python check_dataset.py."""

from pathlib import Path

import yaml

from ultralytics.data.utils import IMG_FORMATS

ROOT = Path(__file__).parent
DATA = ROOT / "ultralytics" / "cfg" / "datasets" / "robo.yaml"

d = yaml.safe_load(DATA.read_text(encoding="utf-8"))
base = Path(d["path"])
print(f"data yaml : {DATA}")
print(f"dataset   : {base}")
print(f"names     : {d['names']}")

ok = True
for split in ("train", "val"):
    img_dir = base / d[split]
    lbl_dir = base / "labels" / split
    if not img_dir.is_dir():
        print(f"[FAIL] {split}: 图片目录不存在 {img_dir}")
        ok = False
        continue
    imgs = [f for f in img_dir.iterdir() if f.suffix.lower().lstrip(".") in IMG_FORMATS]
    lbls = list(lbl_dir.glob("*.txt")) if lbl_dir.is_dir() else []
    stems_img = {p.stem for p in imgs}
    stems_lbl = {p.stem for p in lbls}
    no_label = sorted(stems_img - stems_lbl)
    no_img = sorted(stems_lbl - stems_img)
    empty = [p.name for p in lbls if p.stat().st_size == 0]
    print(f"\n[{split}] images={len(imgs)} labels={len(lbls)}")
    print(f"  无标签图片(按背景处理) {len(no_label)}: {no_label[:8]}")
    print(f"  孤儿标签(无对应图) {len(no_img)}: {no_img[:8]}")
    print(f"  空标签文件 {len(empty)}: {empty[:8]}")
    if not imgs:
        ok = False
        print(f"[FAIL] {split}: 没有可识别格式的图片")

    # 标签边界自检
    bad = []
    nc = len(d["names"])
    for p in lbls:
        for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            s = line.split()
            if not s:
                continue
            if len(s) != 5:
                bad.append(f"{p.name}:{ln} 字段数={len(s)}")
                continue
            c = int(float(s[0]))
            vals = [float(v) for v in s[1:]]
            if c < 0 or c >= nc:
                bad.append(f"{p.name}:{ln} class={c} 超出 0..{nc - 1}")
            if any(v < 0 or v > 1 for v in vals):
                bad.append(f"{p.name}:{ln} 坐标越界 {vals}")
    print(f"  标签异常行 {len(bad)}: {bad[:6]}")
    if bad:
        ok = False

print("\n=== 结论:", "可以训练" if ok else "需修复问题", "===")
