# FactoryBN / BNPDFormer

Bottleneck prediction for HC Factory, adapted from:

- **PDFormer** (Jiang et al., AAAI 2023) — propagation-delay-aware ST Transformer
- **ST-GNN Point Process** (Jin et al., 2023) — congestion / bottleneck *event* prediction

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
```

Artifacts:

| Path | Role |
|------|------|
| `raw_data/<tag>/episodes.npz` | Primary training tensors |
| `raw_data/<tag>/FactoryBN.geo/.rel/.dyna` | Optional LibCity atomic files |
| `libcity/cache/model_cache/<tag>/BNPDFormer_best.pt` | Best checkpoint |

See repo root doc: `06.瓶颈预测模型_PDFormer与点过程适配.md`.
See borrowing map: `factory_bn/BORROWING.md`.
