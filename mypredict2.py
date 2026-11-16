from ultralytics import YOLO

model = YOLO(r"D:\DeepLearning\ultralytics-8.4.202\ultralytics-8.4.202\runs\detect\train-3\weights\best.pt")
model.predict(
    source=r"D:\frames\下载 (2)",
    save=True,
    show=False,
    save_txt=True,
    conf=0.6,
)