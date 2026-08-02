The dataset link is [Google Drive](https://drive.google.com/drive/folders/176Uogr_kty02NQcM9gB2ZT_ngulEhb0H?usp=share_link). You can download the datasets and place them in the `raw_data` directory.

All 6 datasets come from the [LibCity](https://github.com/LibCity/Bigscity-LibCity) repository, which are processed into the [atomic files](https://bigscity-libcity-docs.readthedocs.io/en/latest/user_guide/data/atomic_files.html) format. The only difference with the datasets provided by origin LibCity repository [here](https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe?usp=sharing) is that the filename of the datasets are differently.

## FactoryBN (HC Factory bottleneck)

Export from Stage-C derived tables:

```bash
python -m factory_bn.export_dataset \
  --run_dir ../output/bottleneck_dataset/18_materials \
  --window_size 60
```

This creates `raw_data/FactoryBN/` (`episodes.npz` + optional LibCity `.geo/.rel/.dyna`).
See `factory_bn/README.md` and repo doc `06.瓶颈预测模型_PDFormer与点过程适配.md`.
