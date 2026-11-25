# 训练权重说明

基于 Ultralytics YOLO11n，自建机器人目标检测数据集（`datasets/robot/data.yaml`），训练参数 `imgsz=800`、`epochs=150`。

| 文件        | 来源                | mAP50  | mAP50-95   | 说明                                |
| ----------- | ------------------- | ------ | ---------- | ----------------------------------- |
| `best.pt`   | runs/detect/train-2 | 0.9880 | **0.8394** | 精度最高，推荐使用                  |
| `latest.pt` | runs/detect/train-6 | 0.9879 | 0.8332     | 最新一轮（146 epoch），精度基本持平 |

配套文件：`args_train2.yaml` / `args_train6.yaml` 为训练参数，`results_train2.csv` 为逐 epoch 训练日志。

## 使用方式

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model.predict("test.jpg", imgsz=800, conf=0.4)
```

## 复现训练

```bash
yolo detect train model=yolo11n.pt data=datasets/robot/data.yaml imgsz=800 epochs=150
```

数据集已随仓库提供（`datasets/`），共 7 个子集，总计约 418 MB。
