# models/

`install.bat` downloads these separation models here and checks each against its known hash. Nothing in this
folder is part of the repository; the files keep their authors' terms (see `../THIRD_PARTY_NOTICES.md`).

| File | From | sha256 |
|---|---|---|
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt | `5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa` |
| `model_bs_roformer_ep_317_sdr_12.9755.yaml` | https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_bs_roformer_ep_317_sdr_12.9755.yaml | `2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f` |
| `BS-Rofo-SW-Fixed.ckpt` | https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed | `24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e` |
| `BS-Rofo-SW-Fixed.yaml` | https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed | `f9fada9f94e5ba2d2e4600196299459294bc5f532b314c209cc156ac63e4329b` |

The checkpoints are loaded as plain weights (`weights_only`), so nothing in them can run code. If you already have
Ultimate Vocal Remover, its copy of the BS-RoFormer vocal model is found too.

`hf/` (Full install) is Music Flamingo's download cache.
