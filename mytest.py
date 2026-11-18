import libpng_quiet  # noqa: F401  屏蔽 "libpng warning: iCCP" 刷屏（见 libpng_quiet.py）
from ultralytics import YOLO

model = YOLO(r"yolo11n.pt")
print(model.task)
print(model.names)
print(sum(p.numel() for p in model.parameters()))
