from pathlib import Path

from ultralytics import YOLO

ROOT = Path(r"D:\DeepLearning\ultralytics-8.4.202\ultralytics-8.4.202")
BEST = ROOT / "runs/detect/train-2/weights/best.pt"

if __name__ == "__main__":
    model = YOLO(str(BEST))
    model.train(
        data="robo.yaml",
        epochs=40,
        imgsz=640,
        batch=16,
        cache=False,
        workers=0,
        optimizer="AdamW",
        lr0=0.0002,
        lrf=0.1,
        warmup_epochs=1.0,
        patience=15,
        close_mosaic=5,
        seed=0,
        name="finetune",
        exist_ok=True,
    )
