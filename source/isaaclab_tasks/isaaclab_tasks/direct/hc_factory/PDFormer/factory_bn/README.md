# FactoryBN / BNPDFormer

Bottleneck prediction for HC Factory, adapted from:

- **PDFormer** (Jiang et al., AAAI 2023) — propagation-delay-aware ST Transformer
- **ST-GNN Point Process** (Jin et al., 2023) — kept in the repo, **off** (`use_stgnpp=false`)

Default live head: last **30×60s** windows + `jobs_remaining` → per-station event (`will_block` / `start_min` / `duration_min`) for the next **15 min**, plus `remain_len` and a 6-class process cause. Occupancy grid is auxiliary. Checkpoint metric on the current recipe is **`report_f1`** (station match and start error ≤ 3 min). Train/val/test is an **episode** split, stratified by run prefix.

Metrics contract for baselines: repo-root [`模型评估指标.md`](../../../../../../../模型评估指标.md).

## Quick start

```bash
cd source/isaaclab_tasks/isaaclab_tasks/direct/hc_factory/PDFormer

# 1) Export derived bottleneck tables → raw_data/<tag>
python -m factory_bn.export_dataset \
  --run_dir ../output/bottleneck_dataset/<run_id> \
  --out_dir raw_data/<tag> \
  --window_size 60

# 2) Train (current recipe)
python -m factory_bn.train \
  --config factory_bn/configs/FactoryBN_dense_f1_p80.json \
  --data_dir raw_data/<tag> \
  --save_dir libcity/cache/model_cache/<tag> \
  --max_epoch 30

# 3) Infer (30×60s + remaining jobs → 15 min station events)
python -m factory_bn.infer \
  --ckpt libcity/cache/model_cache/<tag>/BNPDFormer_best.pt \
  --run_dir ../output/bottleneck_dataset/<run_id> \
  --episode 0
```

The only shipped recipe is `FactoryBN_dense_f1_p80.json` (`ckpt_metric=report_f1`, P≥0.80).

`--at last` (default) forecasts after the last observed 60s table.
`--at all` replays every causal step. Live code path: `BNPredictor.predict_x`.
While Isaac is collecting: `python -m bn_agg --follow` then this infer CLI (repo root `08.边采边聚合与在线推理.md`).

Artifacts:

| Path | Role |
|------|------|
| `raw_data/<tag>/episodes.npz` | Primary training tensors (must include `jobs_remaining`) |
| `raw_data/<tag>/FactoryBN.geo/.rel/.dyna` | Optional LibCity atomic files |
| `libcity/cache/model_cache/<tag>/BNPDFormer_best.pt` | Best **report_f1** checkpoint (current recipe) |
| `libcity/cache/model_cache/<tag>/last_metrics.json` | Full test metric dict |

See repo root: `06.PDFormer+STGNPP实现.md`, `04.期望输出.md`, `模型评估指标.md`.
See borrowing map: `factory_bn/BORROWING.md`.
