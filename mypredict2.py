from ultralytics import YOLO

model = YOLO(r"D:\DeepLearning\ultralytics-8.4.202\runs\detect\train-5\weights\best.pt")
model.predict(
    source=r"D:\frames\SECS-4",
    save=True,
    show=False,
    save_txt=True,
    conf=0.6,
)