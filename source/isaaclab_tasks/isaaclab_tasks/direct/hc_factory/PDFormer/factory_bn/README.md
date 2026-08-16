# FactoryBN / BNPDFormer

Bottleneck prediction for HC Factory, adapted from:

- **PDFormer** (Jiang et al., AAAI 2023) — propagation-delay-aware ST Transformer
- **ST-GNN Point Process** (Jin et al., 2023) — congestion / bottleneck *event* prediction

Default head (`remain_to_jobs_done=true`): from the last 12×60s windows + `jobs_remaining`, forecast occupancy until remaining jobs finish (A.1: start / duration / station). Checkpoint metric is **`hot_f1`**. Current `evt/BNPDFormer_best.pt` already has this head; quality is not yet a usable A.1 (see repo `04.期望输出.md`).

## Quick start

```bash
cd source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer

# 1) Export derived bottleneck tables → raw_data/<tag>
python -m factory_bn.export_dataset \
  --run_dir ../output/bottleneck_dataset/old_machine2.0 \
  --out_dir raw_data/old2.0 \
  --window_size 60

# 2) Train multi-task BNPDFormer
python -m factory_bn.train \
  --config factory_bn/configs/FactoryBN.json \
  --data_dir raw_data/old2.0 \
  --save_dir libcity/cache/model_cache/old2.0 \
  --max_epoch 50

# 3) Infer (12×60s windows + remaining jobs → occupancy until jobs done)
python -m factory_bn.infer \
  --ckpt libcity/cache/model_cache/evt/BNPDFormer_best.pt \
  --run_dir ../output/bottleneck_dataset/old_machine2.0 \
  --episode 0
```

`--at last` (default) forecasts remaining occupancy after the last observed 60s table.
`--at all` replays every causal step. Live code path: `BNPredictor.predict_x`.
While Isaac is collecting: `python -m bn_agg --follow` then this infer CLI (repo root `08.边采边聚合与在线推理.md`).

Artifacts:

| Path | Role |
|------|------|
| `raw_data/<tag>/episodes.npz` | Primary training tensors (must include `jobs_remaining`) |
| `raw_data/<tag>/FactoryBN.geo/.rel/.dyna` | Optional LibCity atomic files |
| `libcity/cache/model_cache/<tag>/BNPDFormer_best.pt` | Best **hot_f1** checkpoint |

See repo root docs: `06.PDFormer+STGNPP实现.md`, `04.期望输出.md`.
See borrowing map: `factory_bn/BORROWING.md`.
