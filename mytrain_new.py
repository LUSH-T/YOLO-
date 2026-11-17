import torch
from ultralytics import YOLO
if __name__ =="__main__":
    model=YOLO(r"D:\DeepLearning\ultralytics-8.4.202\ultralytics-8.4.202\runs\detect\train-3\weights\best.pt")
    model.train(
        data="robot.yaml",  # 数据集配置，指向 datasets/robo 的 images 和 labels
        imgsz=800,  # 输入分辨率，每张图先缩放到 640×640 再送进网络
        batch=-1,  # 每批 8 张。原 -1 自动探到 25，验证阶段申请不下内存
        cache="ram",  # 不缓存图片到内存。原 "ram" 多吃 0.3GB 且不确定
        workers=0,  # 数据加载进程数，必须 0。Windows 多进程会崩 (WinError 1455)
        amp=True,  # 混合精度训练，省显存、提速，精度基本不掉
        epochs=150,
        patience=30,# 连续 30 轮 mAP 不涨就提前停，不白烧时间
        #optimizer="AdamW",  # 显式指定优化器。不写的话 auto 会忽略下面的 lr0
        #lr0=0.01,  # 初始学习率。从已有 best.pt 微调，步子要小
        #cos_lr=True,  # 余弦衰减，学习率后期逐渐变小，收敛更稳
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

