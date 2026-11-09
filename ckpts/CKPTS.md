# Checkpoint manifest

Downloaded or copied on the dates below. Every artifact below was checked
with `file` and `sha256sum`; none was HTML.  The checkpoint files were treated
as untrusted data: they were not unpickled, loaded, extracted, or executed.

A02 and A04 use CIFAR-100 checkpoints. Dates use America/Chicago time.

## A02 — STEP Spikformer-4-384

| file | source | bytes | sha256 | file check | date |
|---|---|---:|---|---|---|
| `A02/model_best.pth.tar` | <https://huggingface.co/Fancysean/STEP/resolve/main/ckpt/cls/Spikformer/cifar100/model_best.pth.tar> | 112,620,152 | `e4449e1a7a3271a52e2be4d3dbb90a3baac13e1cf6394c2e9ed7f0ef68d66dbf` | Zip archive data; not HTML | 2026-11-06 |

The loader now reads the CIFAR-100 checkpoint below (Atlas A02); the CIFAR-10 file above is Atlas A01 and is no longer loaded.

| file | source | bytes | sha256 | file check | date |
|---|---|---:|---|---|---|
| `A02/cifar100/model_best.pth.tar` | <https://huggingface.co/Fancysean/STEP/resolve/main/ckpt/cls/Spikformer/cifar100/model_best.pth.tar> | 112,620,152 | `e4449e1a7a3271a52e2be4d3dbb90a3baac13e1cf6394c2e9ed7f0ef68d66dbf` | Zip archive data; not HTML | 2026-11-09 |

```bash
mkdir -p ckpts/A02/cifar100 && curl -L -o ckpts/A02/cifar100/model_best.pth.tar \
  https://huggingface.co/Fancysean/STEP/resolve/main/ckpt/cls/Spikformer/cifar100/model_best.pth.tar
```

## A04 — STEP Spike-driven Transformer (SDT) 2-512

| file | source | bytes | sha256 | file check | date |
|---|---|---:|---|---|---|
| `A04/model_best.pth.tar` | <https://huggingface.co/Fancysean/STEP/resolve/main/ckpt/cls/SDT/cifar100/model_best.pth.tar> | 123,547,800 | `13a5e878b0d21590d7091efdb0d3d8b3759ce5134cc3ce83b7117959bf5b662d` | Zip archive data; not HTML | 2026-11-06 |

The loader now reads the CIFAR-100 checkpoint below (Atlas A04); the CIFAR-10 file above is Atlas A03 and is no longer loaded.

| file | source | bytes | sha256 | file check | date |
|---|---|---:|---|---|---|
| `A04/cifar100/model_best.pth.tar` | <https://huggingface.co/Fancysean/STEP/resolve/main/ckpt/cls/SDT/cifar100/model_best.pth.tar> | 123,547,800 | `13a5e878b0d21590d7091efdb0d3d8b3759ce5134cc3ce83b7117959bf5b662d` | Zip archive data; not HTML | 2026-11-09 |

```bash
mkdir -p ckpts/A04/cifar100 && curl -L -o ckpts/A04/cifar100/model_best.pth.tar \
  https://huggingface.co/Fancysean/STEP/resolve/main/ckpt/cls/SDT/cifar100/model_best.pth.tar
```

## A07 — SpikingLM

| file | source | bytes | sha256 | file check | date |
|---|---|---:|---|---|---|
| `A07/model.safetensors` | <https://huggingface.co/hamingsi/SpikingLM/resolve/main/model.safetensors> | 438,197,096 | `5a591db6049da28c2356ca56b61ffb6ee5bd5a8fe5ce3d4d2f3b3722a81373b7` | generic data; not HTML | 2026-11-06 |
| `A07/config.json` | <https://huggingface.co/hamingsi/SpikingLM/resolve/main/config.json> | 628 | `8e089eb159099dc848c8ab0c6c741bb746198be3c25dfc58808a3667b8550dc0` | JSON text; not HTML | 2026-11-06 |
| `A07/tokenizer.json` | <https://huggingface.co/hamingsi/SpikingLM/resolve/main/tokenizer.json> | 711,396 | `d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66` | JSON text; not HTML | 2026-11-06 |
| `A07/tokenizer_config.json` | <https://huggingface.co/hamingsi/SpikingLM/resolve/main/tokenizer_config.json> | 1,221 | `1f92c93d2e31974acffab86d48e80a9357acb18c68346c5472ef1d12c72d5d77` | JSON text; not HTML | 2026-11-06 |
| `A07/special_tokens_map.json` | <https://huggingface.co/hamingsi/SpikingLM/resolve/main/special_tokens_map.json> | 125 | `b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3` | JSON text; not HTML | 2026-11-06 |
| `A07/vocab.txt` | <https://huggingface.co/hamingsi/SpikingLM/resolve/main/vocab.txt> | 231,508 | `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3` | UTF-8 text; not HTML | 2026-11-06 |

The model size matches the expected 438,197,096 bytes.

## A08 — Prosperity

Pinned workload: Prosperity `vgg16_cifar10_loas`, reusing
`ckpts/B09/vgg16_final_dict.pth.tar` (58,988,519 bytes; SHA-256
`97c83b528447b58164e649622576a07886531559348c42ff63b11eac54481f9f`).  No
new A08 checkpoint was downloaded.

## B09 — LoAS

These files were copied byte-for-byte from the committed upstream clone at
`upstream/LoAS/sample_ckpts/`.

| file | source path | bytes | sha256 | file check | date |
|---|---|---:|---|---|---|
| `B09/vgg16_final_dict.pth.tar` | `upstream/LoAS/sample_ckpts/vgg16_final_dict.pth.tar` | 58,988,519 | `97c83b528447b58164e649622576a07886531559348c42ff63b11eac54481f9f` | Zip archive data; not HTML | 2026-11-06 |
| `B09/resnet19_final_dict.pth.tar` | `upstream/LoAS/sample_ckpts/resnet19_final_dict.pth.tar` | 50,628,133 | `609d7955f4b44d0db5d92f6b2587f72206612e2284c5dc06a544f80c07301a88` | Zip archive data; not HTML | 2026-11-06 |
| `B09/alexnet_final_dict.pth.tar` | `upstream/LoAS/sample_ckpts/alexnet_final_dict.pth.tar` | 9,051,137 | `8f96a44fcfe4c67ef44f3c0837aa2458e22fec8c7a94b7b7051983698865b492` | Zip archive data; not HTML | 2026-11-06 |

## C01 — SMA-AZO-VGG

The two user-pinned SMA-AZO-VGG checkpoints were downloaded with `gdown` from
the exact Google Drive links in `upstream/SMA-AZO/README.md`.  These two files
have no published size or digest to corroborate against (Drive HEAD serves
HTML), unlike the Hugging Face rows above.  Four other C01 model/dataset rows
exist upstream, and SMA-VGG/CIFAR10-DVS has a dead link; see
`upstream/INVENTORY.md`.

| file | source | bytes | sha256 | file check | date |
|---|---|---:|---|---|---|
| `C01/SMA-AZO-VGG_DVS128-Gesture.pth` | <https://drive.google.com/file/d/1YhRPXQQlWmXr5404j_RjiHtvlarHWyPx/view?usp=drive_link> | 89,886,835 | `0701c30e349eed358a2b146e8373431f384674941259ce85491c0949a0076591` | Zip archive data; not HTML | 2026-11-06 |
| `C01/SMA-AZO-VGG_CIFAR10-DVS.pth` | <https://drive.google.com/file/d/1zXzSUedGMBkqiCIQNKvLPQxoCaDi-uGQ/view?usp=drive_link> | 89,881,567 | `1f6dd922f00c898f9ec1111d7cdffb83dcda942a7c6d2418aeafac58dc02dc31` | Zip archive data; not HTML | 2026-11-06 |
