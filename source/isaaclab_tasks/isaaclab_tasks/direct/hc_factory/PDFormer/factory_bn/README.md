# FactoryBN / BNPDFormer

Bottleneck prediction for HC Factory, adapted from:

- **PDFormer** (Jiang et al., AAAI 2023) — propagation-delay-aware ST Transformer
- **ST-GNN Point Process** (Jin et al., 2023) — congestion / bottleneck *event* prediction

## Quick start

```bash
cd source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer

# 1) Export derived bottleneck tables → raw_data/FactoryBN
python -m factory_bn.export_dataset \
  --run_dir ../output/bottleneck_dataset/18_materials \
  --window_size 60

# 2) Train multi-task BNPDFormer
python -m factory_bn.train \
  --config factory_bn/configs/FactoryBN.json \
  --max_epoch 50
```

Artifacts:

| Path | Role |
|------|------|
| `raw_data/FactoryBN/episodes.npz` | Primary training tensors |
| `raw_data/FactoryBN/*.geo/.rel/.dyna` | Optional LibCity atomic files |
| `libcity/cache/model_cache/FactoryBN/BNPDFormer_best.pt` | Best checkpoint |

See repo root doc: `06.瓶颈预测模型_PDFormer与点过程适配.md`.
See borrowing map: `factory_bn/BORROWING.md`.
