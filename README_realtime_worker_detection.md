# Realtime Worker Detection 运行说明

本说明用于运行实时工人检测脚本 `realtime_worker_detect.py`。脚本会持续检查输入图片文件夹：有新图片就处理；没有新图片就等待；连续多次没有新图片后自动结束。

---

## 1. 当前路径

### GroundingDINO 代码目录

```bash
/home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main
```

### 输入图片目录

```bash
/home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/3orders_final_debug_camera
```

### 输出结果目录

```bash
/home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/outputs_3orders_final_worker_detection_realtime
```

---

## 2. 进入运行环境

每次新开终端后，先执行：

```bash
conda activate /home/sci/work/zhw_envs/dino_worker

cd /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main

source setup_dino_env.sh
```

检查当前目录：

```bash
pwd
```

应输出：

```bash
/home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main
```

检查脚本、配置文件和权重是否存在：

```bash
ls -lh realtime_worker_detect.py
ls -lh groundingdino/config/GroundingDINO_SwinT_OGC.py
ls -lh weights/groundingdino_swint_ogc.pth
```

检查 GroundingDINO 是否可以正常导入：

```bash
python -c "from groundingdino.util.inference import load_model, load_image, predict; print('GroundingDINO inference OK')"
```

正常输出：

```text
GroundingDINO inference OK
```

---

## 3. 实时检测命令

在 `GroundingDINO-main` 目录下运行：

```bash
python realtime_worker_detect.py \
  --input_dir /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/3orders_final_debug_camera \
  --output_dir /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/outputs_3orders_final_worker_detection_realtime \
  --config /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --weights /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main/weights/groundingdino_swint_ogc.pth \
  --prompt "person . worker . human ." \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --poll_interval 5 \
  --max_empty_rounds 12 \
  --min_file_age 1 \
  --save_annotated
```

该命令含义：

```text
每 5 秒检查一次输入文件夹；
如果发现新图片，就处理这些新图片；
如果没有新图片，就等待 5 秒后再次检查；
连续 12 次没有新图片后自动结束；
也就是连续约 60 秒没有新图片后程序退出。
```

---

## 4. 只处理启动后新生成的图片

如果输入文件夹里已经有旧图片，但只想处理程序启动后新产生的图片，可以加上：

```bash
--ignore_existing
```

完整命令：

```bash
python realtime_worker_detect.py \
  --input_dir /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/3orders_final_debug_camera \
  --output_dir /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/outputs_3orders_final_worker_detection_realtime \
  --config /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --weights /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main/weights/groundingdino_swint_ogc.pth \
  --prompt "person . worker . human ." \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --poll_interval 5 \
  --max_empty_rounds 12 \
  --min_file_age 1 \
  --save_annotated \
  --ignore_existing
```

适用场景：仿真程序正在实时生成图片，检测程序只需要从启动时刻开始处理新图片。

---

## 5. 后台运行

如果希望程序在后台运行，使用 `nohup`：

```bash
nohup python realtime_worker_detect.py \
  --input_dir /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/3orders_final_debug_camera \
  --output_dir /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/outputs_3orders_final_worker_detection_realtime \
  --config /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --weights /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main/weights/groundingdino_swint_ogc.pth \
  --prompt "person . worker . human ." \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --poll_interval 5 \
  --max_empty_rounds 12 \
  --min_file_age 1 \
  --save_annotated \
  > run_worker_detection_final_realtime.log 2>&1 &
```

查看日志：

```bash
tail -f run_worker_detection_final_realtime.log
```

退出日志查看：

```bash
Ctrl + C
```

注意：`Ctrl + C` 只会退出日志查看，不会停止后台检测程序。

查看程序是否仍在运行：

```bash
ps -ef | grep realtime_worker_detect.py
```

---

## 6. 输入数据格式

输入目录可以包含多个相机子文件夹，例如：

```text
3orders_final_debug_camera/
├── env_00_camera_xxx/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── env_00_camera_yyy/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
└── ...
```

脚本会递归扫描所有子文件夹中的图片。

支持的图片格式：

```text
.jpg
.jpeg
.png
.bmp
.webp
```

---

## 7. 输出文件

运行后，输出目录中会生成：

```text
outputs_3orders_final_worker_detection_realtime/
├── results.csv
├── results.json
├── results.jsonl
├── processed_images.txt
├── failed_images.txt
└── annotated/
```

### 7.1 results.csv

CSV 格式检测结果，每一行对应一张图片中的一个检测目标。

主要字段：

| 字段 | 含义 |
|---|---|
| image_path | 原始图片完整路径 |
| relative_path | 相对路径，包含相机文件夹和图片名 |
| has_worker | 当前图片是否检测到工人 |
| num_workers | 当前图片检测到的工人数 |
| worker_index | 当前图片中的检测目标序号 |
| worker_id | 工人编号，例如 worker_0 |
| color | 识别出的衣服颜色 |
| det_score | GroundingDINO 检测匹配分数 |
| color_confidence | 颜色识别置信度 |
| x1, y1, x2, y2 | 检测框坐标 |
| phrase | GroundingDINO 匹配到的文本词 |

查看前 20 行：

```bash
head -20 /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/outputs_3orders_final_worker_detection_realtime/results.csv
```

### 7.2 results.json

普通 JSON 格式结果，便于后处理程序读取。每张图片对应一个对象，主要包括：

```text
image_path
relative_path
has_worker
num_workers
detections
annotated_path
```

### 7.3 results.jsonl

实时追加写入的 JSONL 文件，每一行是一张图片的检测结果。即使程序中途停止，已经处理过的结果也不会丢失。

### 7.4 processed_images.txt

记录已经处理过的图片相对路径，用于避免重复处理同一张图片。

### 7.5 failed_images.txt

记录处理失败的图片和错误信息。

### 7.6 annotated/

保存带检测框和标签的可视化图片。

图片标签示例：

```text
worker_3 | yellow | det=0.26 | color=0.74
```

含义：

```text
worker_3：识别为 3 号工人
yellow：衣服颜色识别为黄色
det=0.26：GroundingDINO 检测匹配分数
color=0.74：颜色识别置信度
```

---

## 8. 工人编号规则

当前工人编号由衣服颜色决定：

| 衣服颜色 | 工人编号 |
|---|---|
| red | worker_0 |
| green | worker_1 |
| light_blue | worker_2 |
| yellow | worker_3 |
| dark_blue | worker_4 |
| unknown | unknown |

---

## 9. 参数说明

### `--poll_interval`

检查新图片的时间间隔，单位是秒。

```bash
--poll_interval 5
```

表示每 5 秒检查一次输入文件夹。

### `--max_empty_rounds`

连续多少次没有新图片后结束程序。

```bash
--max_empty_rounds 12
```

配合 `--poll_interval 5`，表示连续约 60 秒没有新图片后程序结束。

### `--min_file_age`

判断图片是否写入完成的等待时间。

```bash
--min_file_age 1
```

表示图片最后修改时间距离当前至少 1 秒，才认为文件已经写完，可以处理。

如果图片较大，或者仿真程序写图速度慢，可以改成：

```bash
--min_file_age 2
```

### `--box_threshold`

检测框保留阈值。

```bash
--box_threshold 0.25
```

推荐：

```text
漏检较多：降低到 0.20
误检较多：提高到 0.30 或 0.35
```

### `--text_threshold`

文本匹配阈值。

```bash
--text_threshold 0.25
```

推荐：

```text
漏检较多：降低到 0.20
误检较多：提高到 0.30
```

### `--save_annotated`

保存带检测框的可视化图片。如果只需要 `results.csv` 和 `results.json`，可以去掉该参数。

### `--ignore_existing`

启动时忽略输入文件夹中已经存在的图片，只处理启动后新增的图片。

---

## 10. 常见问题

### 报错：`libc10.so` 找不到

一般是环境变量没有加载。重新执行：

```bash
source setup_dino_env.sh
```

### 报错：`No module named groundingdino`

确认当前目录是否为：

```bash
/home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main
```

然后重新执行：

```bash
source setup_dino_env.sh
```

### 没有检测到图片

检查输入目录是否正确：

```bash
find /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/3orders_final_debug_camera \
  -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | head
```

统计图片数量：

```bash
find /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/3orders_final_debug_camera \
  -type f \( -iname "*.jpg" -o -iname "*.png" -o -iname "*.jpeg" \) | wc -l
```

### 程序很快结束

当前设置是连续约 60 秒没有新图片后结束：

```bash
--poll_interval 5
--max_empty_rounds 12
```

如果希望等待更久，可以改成：

```bash
--max_empty_rounds 60
```

表示约 5 分钟没有新图片后结束。

### 中途停止后重新运行会重复处理吗

默认不会重复处理已经记录在 `processed_images.txt` 中的图片。

如果想重新完整处理一次，需要删除旧输出目录，或者换一个新的 `--output_dir`。

例如：

```bash
rm -rf /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/outputs_3orders_final_worker_detection_realtime
```

然后重新运行检测命令。

---

## 11. 推荐运行流程

```bash
# 1. 进入环境
conda activate /home/sci/work/zhw_envs/dino_worker

# 2. 进入 GroundingDINO 目录
cd /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main

# 3. 加载环境变量
source setup_dino_env.sh

# 4. 检查导入
python -c "from groundingdino.util.inference import load_model, load_image, predict; print('GroundingDINO inference OK')"

# 5. 运行实时检测
python realtime_worker_detect.py \
  --input_dir /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/3orders_final_debug_camera \
  --output_dir /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/outputs_3orders_final_worker_detection_realtime \
  --config /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --weights /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main/weights/groundingdino_swint_ogc.pth \
  --prompt "person . worker . human ." \
  --box_threshold 0.25 \
  --text_threshold 0.25 \
  --poll_interval 5 \
  --max_empty_rounds 12 \
  --min_file_age 1 \
  --save_annotated
```

---

## 12. 当前脚本阶段说明

当前实时检测脚本负责：

```text
1. 监控输入文件夹是否出现新图片；
2. 对新增图片运行 GroundingDINO 工人检测；
3. 根据衣服颜色输出 worker_id；
4. 生成 results.csv / results.json / results.jsonl；
5. 可选保存 annotated 标注图；
6. 连续一段时间没有新图片后自动结束程序。
```

按时间戳聚合“几号工人在哪个机器旁边”的后处理逻辑暂时先不在该脚本中执行，等相机视角最终确定后再统一处理。
