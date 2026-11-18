r"""
增量迭代训练：用「新数据集」+「上一轮训好的权重」继续训练。.

用法：
    1) 先把新图片+同名标注放进 datasets\robot\\images\train 与 labels\train
    2) 删掉 labels\\*.cache（或跑 python pretrain_check.py --clean-cache）
    3) Ctrl+F5 运行本脚本（不要用 F5 调试启动）

两种模式改一行即可：
    MODE = "warm"   从 best.pt 接着训（微调，推荐）
    MODE = "resume" 断点续训上一轮未跑完的任务（连优化器/学习率/epoch 一起恢复）
"""

from ultralytics import YOLO

# ---------------- 配置 ----------------
MODE = "warm"

PREV_BEST = r"D:\DeepLearning\ultralytics-8.4.202\runs\detect\train-3\weights\best.pt"  # 上一轮最好权重
PREV_LAST = r"D:\DeepLearning\ultralytics-8.4.202\runs\detect\train-3\weights\last.pt"  # 上一轮最后权重（resume 用）
DATA = r"D:\DeepLearning\ultralytics-8.4.202\ultralytics-8.4.202\datasets\robot\data.yaml"

# ---------------- 训练 ----------------
if __name__ == "__main__":
    if MODE == "resume":
        # 续训：必须用 last.pt，且不要传其它超参，否则 resume 失效
        model = YOLO(PREV_LAST)
        model.train(resume=True)
    else:
        model = YOLO(PREV_BEST)  # 只继承权重，不继承优化器状态 → 全新一轮
        model.train(
            data=DATA,
            # ---------- 数据 ----------
            imgsz=800,
            batch=-1,  # 自动探显存
            cache="ram",
            workers=0,  # Windows 多进程会崩，保持 0
            amp=True,
            # ---------- 时长 ----------
            epochs=40,  # 微调不需要 150 轮，30~50 足够
            patience=15,  # 15 轮不涨就停
            # ---------- 学习率（关键）----------
            optimizer="AdamW",
            lr0=5e-4,  # 从已有权重出发，必须比从头训的 0.01 小一个量级
            lrf=0.05,  # 末期学习率 = lr0 * 0.05
            cos_lr=True,
            warmup_epochs=1,  # 已收敛的权重不需要长 warmup
            # ---------- 增强（比从头训温和一些）----------
            degrees=10,
            scale=0.5,
            mosaic=0.5,  # 微调期 mosaic 降一点，避免把已学好的语义搅乱
            close_mosaic=10,
            fliplr=0.5,
            hsv_v=0.5,
            erasing=0.3,
            mixup=0.0,  # 微调阶段关掉 mixup/cutmix，收敛更稳
            cutmix=0.0,
        )
