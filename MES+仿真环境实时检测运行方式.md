# 运行方式

## IsaacSIM仿真端：

1. 终端1：启动仿真环境

```
conda activate env_isaaclab
python train.py \
  --task HRTPaHC-v1 \
  --algo rule_based \
  --num_envs 1 \
  --active_livestream \
  --livestream_public_ip 10.68.14.234 \
  --livestream_port 49100 \
  --device cuda:0 \
  --headless
  --enable_camera
```

1. 终端2：暴露图片文件夹端口：

```
cd ~/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output
python3 -m http.server 8888 --bind 0.0.0.0
```

1. 终端3：运行dino实时检测：

```bash
1. 进入虚拟环境
conda activate /home/sci/work/zhw_envs/dino_worker

cd /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main

source setup_dino_env.sh
2. 运行实时检测(具体参数含义见README_realtime_worker_detection.md)：
python realtime_worker_detect.py \
  --input_dir /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/debug_camera \
  --output_dir /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/output/worker_detection_realtime \
  --config /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --weights /home/sci/work/dino_isaac_factory/source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/GroundingDINO-main/weights/groundingdino_swint_ogc.pth \
  --prompt "person . worker . human ." \
  --box_threshold 0.33 \
  --text_threshold 0.33 \
  --poll_interval 5 \
  --max_empty_rounds 12 \
  --min_file_age 1 \
  --save_annotated
```



## MES前端：

```
pnpm dev
```

