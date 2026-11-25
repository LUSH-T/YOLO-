from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"D:\DeepLearning\ultralytics-8.4.202\runs\detect\train-5\weights\best.pt")
    model.train(
        data="robot.yaml",
        imgsz=800,
        batch=-1,
        cache="ram",
        workers=0,
        amp=True,  # 混合精度训练，省显存、提速，精度基本不掉
        epochs=150,
        patience=30,  # 连续 30 轮 mAP 不涨就提前停，不白烧时间
        optimizer="AdamW",  # 显式指定优化器。不写的话 auto 会忽略下面的 lr0
        lr0=0.001,  # 初始学习率。从已有 best.pt 微调
        cos_lr=True,  # 余弦衰减
        degrees=10,  # 随机旋转
        scale=0.75,  # 随机缩放
        mosaic=0.8,  # 四图拼接
        close_mosaic=20,  # 最后 轮关掉 mosaic
        fliplr=0.5,  # 水平翻转
        hsv_v=0.5,  # 亮度变化
        erasing=0.4,
        mixup=0.1,
        cutmix=0.05,
    )
