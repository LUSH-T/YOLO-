from ultralytics import YOLO

VIDEO = r"D:\视频下载\带着学生从底层开始大面积改进的3D自动建图导航框架终于有不错进展了，宇树GO2真机测试中，不断进化中。开源的3D框架一到真机就没好货。没有免费的午餐，只有钻到底.mp4"

print("[1/3] 脚本已启动", flush=True)
print(f"      视频: {VIDEO}", flush=True)

model = YOLO(r"D:\DeepLearning\ultralytics-8.4.202\ultralytics-8.4.202\runs\detect\train-2\weights\best.pt")
print(f"[2/3] 模型加载完成, 类别: {model.names}", flush=True)

# stream=True: 逐帧返回生成器, 结果不再全部堆积在内存里
# verbose=False: 关掉逐帧日志(注意它也会关掉进度条), 所以下面自己打进度
results = model.predict(
    source=VIDEO,
    save=True,
    show=False,
    save_txt=False,
    conf=0.7,
    stream=True,
    verbose=False,
)

print("[3/3] 开始逐帧推理, 下面会实时输出...", flush=True)

total_frames = 0
hit_frames = 0
for i, r in enumerate(results, 1):
    total_frames = i
    n = len(r.boxes)
    if n:
        hit_frames += 1
        cls_names = " ".join(model.names[int(c)] for c in r.boxes.cls)
        print(f"frame {i}: {n} x [{cls_names}]", flush=True)
    if i % 100 == 0:
        print(f"--- 已处理 {i} 帧, 命中 {hit_frames} 帧 ---", flush=True)

if total_frames:
    print(f"\n完成: 共 {total_frames} 帧, 有检测的帧 {hit_frames} ({hit_frames/total_frames:.1%})")
    print("结果视频保存在 runs/detect/predict* 目录下")
else:
    print("\n一帧都没读到, 检查视频路径和文件是否可解码")
