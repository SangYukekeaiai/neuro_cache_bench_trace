# Upstream benchmark inventory

This is a read-only reconnaissance of the five upstream repositories requested
for benchmark IDs A02/A04/A07/A08/B09/C01.  The repositories under this
directory are untrusted data: no upstream code was run and no upstream
requirements were installed.  Checkpoint availability below is based on
committed files or a `curl -sI -L` probe; no weight file was downloaded.

## A02 — STEP / Spikformer-4-384

### Checkpoint

The pinned configuration is the CIFAR-10 `spikformer_cifar` entry in
`STEP/cls/configs/spikformer/cifar10.yml`:

- `step: 4`, `embed_dim: 384`, `depths: 4`, `num_heads: 12`, `patch_size: 4`,
  `attn_layer: SSA`, and `node_type: LIFNode`.
- The trained checkpoint path in the public Hugging Face repository is
  `ckpt/cls/Spikformer/cifar10/model_best.pth.tar`.
- URL:
  <https://huggingface.co/Fancysean/STEP/resolve/main/ckpt/cls/Spikformer/cifar10/model_best.pth.tar>
- `curl -sI -L` ended at HTTP 200 with `Content-Type: application/x-tar` and
  the public LFS size 112,204,280 bytes.  No credentials were requested, so it
  is publicly downloadable as observed.

### Dataset, resolution, and timestep

The dataset is static `torch/cifar10`, 10 classes, with 32x32 RGB input and
`T=4` direct-encoded timesteps.

### Spiking layers and binary-input audit

The SPS stem has 3x3 stride-1 convolutions (`proj_conv`, `proj_conv1`,
`proj_conv2`, `proj_conv3`, and `rpe_conv`) with LIF nodes.  The first
`proj_conv` sees the normalized continuous CIFAR image, so its input is not
binary.  Convolutions after a LIF see hard spike outputs (0/1).  The final
residual `x + x_feat` can be non-binary even though each branch came from a
LIF node.

SSA has q/k/v and projection linear GEMMs, two attention GEMMs (`q @ k^T` and
`attention @ v`), MLP linear GEMMs, and a final classifier linear layer.  q/k/v
after their LIF nodes are binary, but q/k/v linear inputs are the stem output
including the residual sum and therefore are not guaranteed binary.  Attention
products, the projection input, MLP residual input, and classifier input are
non-binary; an MLP linear immediately after an LIF receives binary input.  The
local `SigmoidGrad`/LIF implementation uses a hard threshold, so the stated
binary points are source-level 0/1 outputs, not merely sparse floating values.

### Pins

`STEP/cls/requirements.txt` pins Python-compatible dependencies including
`torch==2.4.1`, `torchvision==0.19.1`, `torchaudio==2.4.1`, `numpy==1.24.4`,
`timm==0.5.4`, and `spikingjelly==0.0.0.0.14`.  The root README advertises
Python 3.8, 3.9, or 3.10.  BrainCog is an external GitHub prerequisite and is
not version-pinned in the requirements file.

## A04 — STEP / Spike-driven Transformer (SDT) 2-512

### Checkpoint

The pinned configuration is the CIFAR-10 `sdt_cifar` entry in
`STEP/cls/configs/sdt/cifar10.yml`:

- `step: 4`, `embed_dim: 512`, `depths: 2`, `num_heads: 8`, `patch_size: 4`,
  `attn_layer: SDSA`, and `node_type: LIFNode`.  Thus “2-512” denotes the
  two-block/512-dimensional model; the simulation timestep in this config is
  four.
- The trained checkpoint path in the public Hugging Face repository is
  `ckpt/cls/SDT/cifar10/model_best.pth.tar`.
- URL:
  <https://huggingface.co/Fancysean/STEP/resolve/main/ckpt/cls/SDT/cifar10/model_best.pth.tar>
- `curl -sI -L` ended at HTTP 200 with `Content-Type: application/x-tar` and
  the public LFS size 122,993,688 bytes.  No credentials were requested, so it
  is publicly downloadable as observed.

### Dataset, resolution, and timestep

The dataset is static `torch/cifar10`, 10 classes, with 32x32 RGB input and
`T=4` direct-encoded timesteps.

### Spiking layers and binary-input audit

The SPS stem uses 3x3 stride-1 convolutions (`proj_conv` through
`proj_conv3`) and 3x3 `rpe_conv`, with LIF nodes and max pooling.  The first
convolution receives continuous normalized image values.  `proj_conv1` and
`proj_conv2` receive preceding LIF outputs and therefore binary 0/1 inputs.
The source feeds `proj_conv3` through max pooling before its LIF, so that input
is binary from `proj_lif2`; `rpe_conv` instead receives the pre-LIF-3
activation and is not guaranteed binary.  Its residual sum with `x_feat` is
also non-binary in general.

SDSA uses 1x1 q/k/v convolutions and a 1x1 projection convolution, with
elementwise q/k/v products and reductions (`k * v`, sum, then `q * kv`), not
the quadratic q-k attention GEMMs used by Spikformer.  The q/k/v convolution
inputs are after `shortcut_lif` and are binary; batch-normalized values before
each q/k/v LIF are non-binary, and the reductions/projection residual are
non-binary.  The two-block MLP uses 1x1 `fc_conv1`/`fc_conv2` GEMMs: each
convolution input is explicitly passed through an LIF and is binary.  The
final classifier sees the non-binary averaged hidden state.

### Pins

The pins are the same STEP pins listed for A02: Python 3.8/3.9/3.10 in the
README, `torch==2.4.1`, `torchvision==0.19.1`, `torchaudio==2.4.1`,
`numpy==1.24.4`, `timm==0.5.4`, and `spikingjelly==0.0.0.0.14`; BrainCog is
not pinned in the requirements file.

## A07 — SpikingLM

### Checkpoint

No model weights are committed in `SpikingLM`.  The README links the released
checkpoint repository <https://huggingface.co/hamingsi/SpikingLM>, with the
exact model file `model.safetensors` (API metadata size 438,197,096 bytes),
plus `config.json`, tokenizer files, and `vocab.txt`.

- Model URL:
  <https://huggingface.co/hamingsi/SpikingLM/resolve/main/model.safetensors>
- `curl -sI -L` ended at HTTP 200 with `Content-Type: application/octet-stream`
  and no authentication redirect.  The model is publicly downloadable as
  observed; it was not downloaded.

### Dataset, resolution, and timestep

This is a language model, so “resolution” is token sequence length rather than
an image shape.  The final ICML setting uses sequence length 128 and `T=4`.
Pretraining mixes STORIES, BookCorpus, CC-News, OpenWebText, and English
Wikipedia; the repository also supports GLUE fine-tuning.  `configs/bert_base_t4.json`
describes BERT-base dimensions (hidden size 768, 12 layers, 12 heads,
maximum position length 512); the README's reported/pretraining setting is
the 128-token sequence.

### Spiking layers and binary-input audit

There are no convolution layers.  Token IDs are integer inputs to word,
position, and type embeddings; embeddings and residual/LayerNorm states are
continuous, not binary spikes.  Each self-attention block first applies a LIF
(`proj_lif`), then q/k/v linear GEMMs, q/k/v scaling, q/k/v LIFs, and two
attention GEMMs (`query @ key^T` and `attention_probs @ value`).  The linear
inputs immediately after a LIF are binary 0/1, but the scaled linear outputs,
attention scores/probabilities, and context are non-binary.  The attention
output and both MLP dense GEMMs are fed through LIFs at their input, while
LayerNorm residuals and final task heads are continuous.  Thus this model has
binary internal spike points but its raw embedding input and several residual
inputs cannot satisfy a blanket binary assertion.

### Pins

`requirements.txt` does not pin Python or an exact PyTorch version: it only
requires `torch`, `spikingjelly`, and minimum versions for several packages.
The checked-in config records `transformers_version: 4.57.0`; the README does
not specify a Python version.  The repository's own weights are therefore
available, but reproducing the environment needs a user-selected compatible
Python/Torch/CUDA stack.

## A08 — Prosperity

### Checkpoint and scope

Prosperity is a simulator/workload repository, not a model-training release.
No `.pth`, `.pt`, `.ckpt`, or Hugging Face/Google Drive/Zenodo checkpoint URL
was found.  It commits activation-matrix inputs instead, under these exact
paths:

`data/alexnet_cifar10_loas.pkl`, `data/lenet5_mnist.pkl`,
`data/resnet18_cifar10.pkl`, `data/resnet18_cifar100.pkl`,
`data/resnet19_cifar10_loas.pkl`, `data/sdt_cifar10.pkl`,
`data/sdt_cifar100.pkl`, `data/sdt_cifar10dvs.pkl`,
`data/spikebert_mr.pkl`, `data/spikebert_sst2.pkl`,
`data/spikebert_sst5.pkl`, `data/spikformer_cifar10.pkl`,
`data/spikformer_cifar100.pkl`, `data/spikformer_cifar10dvs.pkl`,
`data/spikingbert_mnli.pkl`, `data/spikingbert_qqp.pkl`,
`data/spikingbert_sst2.pkl`, `data/vgg16_cifar10.pkl`,
`data/vgg16_cifar100.pkl`, `data/vgg16_cifar10_loas.pkl`,
`data/vgg9_cifar10.pkl`, and `data/vgg9_cifar10dvs.pkl`.

These are untrusted pickle activation dumps, not downloadable trained
checkpoints.  A trained model must be supplied or trained separately.

### Dataset, resolution, and timestep

A08 is not a single pinned model; the simulator supports the following
configuration families.  The dimensions below are the values in
`simulator/configs.py`, not guesses from the pickle contents.

| family / dataset variants | input resolution or sequence | T |
|---|---:|---:|
| Spikformer / CIFAR-10, CIFAR-100 | 32x32 | 4 |
| Spikformer / CIFAR10-DVS | 64x64 simulator input | 16 |
| SDT / CIFAR-10, CIFAR-100 | 32x32 | 4 |
| SDT / CIFAR10-DVS | 64x64 simulator input | 10 |
| VGG16 / CIFAR activation workloads | 32x32 | 32 |
| VGG9 / CIFAR-10 | 32x32 | 4 |
| VGG9 / CIFAR10-DVS | 128x128 simulator input | 4 |
| LeNet-5 / MNIST | 16x16 simulator feature map | 32 |
| ResNet18 / CIFAR activation workloads | 8x8 simulator input | 32 |
| spikeBERT / SST-2, SST-5 | 128 tokens | 4 |
| spikeBERT / MR | 256 tokens | 4 |
| spikingBERT / SST-2, QQP, MNLI | 128 tokens | 16 |

The repository's simulator dimensions are its effective workload resolution;
the raw dataset or a trainable model is not included for A08.

### Spiking layers and binary-input audit

The simulator models convolution and FC layers as equivalent GEMMs.  Their
activation tensors have dtype `spike`, `nbits=1`; weights and outputs are
modeled as `fp8`.  LIF membrane/input state is `fp8`, while LIF output is
`spike`.  Spikformer and spikeBERT attention include q/k/v spike tensors and
q-k / attention-v GEMMs whose scores/output are `fp8` (non-binary).  SDT's
attention is represented as spike `kv_out`/output with elementwise/reduction
products rather than q-k attention GEMMs.  Thus simulator-marked FC/Conv and
q/k/v inputs are binary by design, while raw inputs, LIF state, attention
accumulations, and fp8 residuals are not guaranteed 0/1.  This is a workload
model, not proof that an absent trained model would emit binary values at
every capture hook.

### Pins

The README recommends Python 3.10 and CUDA >=12.1.  `requirements.txt` pins
`torch==2.3.0`, `numpy==1.26.4`, `matplotlib==3.9.0`, `networkx==3.3`,
`seaborn==0.13.2`, and `pandas==2.2.2`.

## B09 — LoAS

### Checkpoint

LoAS commits trained sample state dictionaries locally; no external checkpoint
URL is provided.  The exact committed paths are:

- `sample_ckpts/alexnet_final_dict.pth.tar` (9,051,137 bytes)
- `sample_ckpts/resnet19_final_dict.pth.tar` (50,628,133 bytes)
- `sample_ckpts/vgg16_final_dict.pth.tar` (58,988,519 bytes)

The corresponding committed profiling matrices are
`artifact/Table-II/src_matrices/alexnet_final_matrices_dict.pth`,
`artifact/Table-II/src_matrices/resnet19_final_matrices_dict.pth`, and
`artifact/Table-II/src_matrices/vgg16_final_matrices_dict.pth`.  These local
files require no credentials or download.  They are state dictionaries, so
the matching architecture source/config is needed to instantiate a model.

### Dataset, resolution, and timestep

`config_profile.py` and `config_ft.py` default to CIFAR-10 at 32x32 RGB,
`T=4`, and expose `vgg16`, `resnet19`, and `alexnet`.  The checked-in sample
checkpoints cover all three architecture names.  CIFAR-100 is exposed by the
lottery-ticket config but is not the profile default.

### Spiking layers and binary-input audit

All three families use convolution layers and final linear/FC layers with
LIF neurons.  VGG16 and AlexNet receive the continuous directly encoded image
at the first convolution; subsequent convolution inputs are outputs of LIF
nodes and are 0/1.  Their final classifier inputs follow binary spiking
paths.  ResNet19 has the same continuous first-convolution input and binary
LIF-to-convolution inputs, but each residual branch adds a BN/shortcut value
before its next LIF, so the pre-LIF residual is non-binary.  ResNet19's first
FC receives pooled spikes, while its second FC receives the non-binary output
of the first FC (there is no LIF between those two FCs).  No attention GEMM is
present in these LoAS CNNs.

### Pins

The README reports Python 3.9.7, CUDA 11.1, PyTorch 2.3.1 (the named conda
build is `py3.9_cuda11.8_cudnn8.7.0_0`), and spikingjelly 0.0.0.0.12.
`requirements.txt` pins `torch==2.3.1`, `torchvision==0.18.1`,
`numpy==1.24.4`, and the listed supporting packages.

## C01 — SMA-AZO

### Checkpoint

No checkpoint files are committed.  The README supplies these Google Drive
links; each `curl -sI -L` probe ended at HTTP 200 without a login redirect.
The final response is Drive HTML (`text/html`), which is a public landing/
download response rather than the weight bytes; no weight was downloaded.
The README explicitly states that these links should work without prior
authorization.

| model / dataset | exact checkpoint URL | credential observation |
|---|---|---|
| SMA-VGG / DVS128 Gesture | <https://drive.google.com/file/d/1IpiNyIzGoFZPB6VadyVDC4NwIjt3KEgX/view?usp=drive_link> | HEAD 200, public link; no auth observed |
| SMA-AZO-VGG / DVS128 Gesture | <https://drive.google.com/file/d/1YhRPXQQlWmXr5404j_RjiHtvlarHWyPx/view?usp=drive_link> | HEAD 200, public link; no auth observed |
| SMA-VGG / CIFAR10-DVS | no URL in README (`[link]()`); unavailable from this repo | not downloadable |
| SMA-AZO-VGG / CIFAR10-DVS | <https://drive.google.com/file/d/1zXzSUedGMBkqiCIQNKvLPQxoCaDi-uGQ/view?usp=drive_link> | HEAD 200, public link; no auth observed |
| SMA-VGG / N-Caltech101 | <https://drive.google.com/file/d/1IBwLOPAtwJBbB7jwg9dB0TdJbMgFoD49/view?usp=drive_link> | HEAD 200, public link; no auth observed |
| SMA-AZO-VGG / N-Caltech101 | <https://drive.google.com/file/d/1-YdptrxyELPF17qpRNVA54miwnsVZTGX/view?usp=drive_link> | HEAD 200, public link; no auth observed |

C01 therefore has multiple trained choices rather than one pinned dataset;
the user must select one of the rows above.  The URLs are exact README URLs.

### Dataset, resolution, and timestep

The training scripts set the following default simulation lengths:

- DVS128 Gesture: `SMA VGG/vgg8_dvs128_SMA.py`, `T=16`, native event-camera
  frames 128x128.
- CIFAR10-DVS: `SMA VGG/vgg8_cifar10dvs_SMA.py`, `T=10`, native event-camera
  frames 128x128.
- N-Caltech101: `SMA VGG/vgg8_NCaltech101_SMA.py`, `T=14`.  The script does
  not resize or declare a shape; it passes the N-Caltech101 native event-frame
  geometry through the SpikingJelly loader (commonly 180x240), so the exact
  capture resolution must be confirmed from the installed dataset loader.

### Spiking layers and binary-input audit

All variants are VGG8-style: 3x3 Conv2d layers (two input event channels),
LIF nodes after the convolution blocks, and linear classifier layers.  The
SMA module adds multiscale spatial Conv2d branches (kernels 1/3/5/7), channel
1x1 Conv2d attention, temporal 1x1 Conv3d attention, softmax weighting, and
nonlinear weighted sums.  It has no q-k / attention GEMM.  LIF outputs are
hard spikes (0/1) and feed subsequent convolution/linear stages, but raw
event-frame input is not asserted binary by these scripts (frame bins can
contain counts).  Batch-normalization, softmax weights, channel/time
attention products, and residual sums are non-binary.  Consequently only the
post-LIF convolution/linear inputs are safe for the capture harness's exact
0/1 assertion.

### Pins

The conda-export `requirements.txt` pins Python 3.9.16, PyTorch 2.0.1
(`py3.9_cuda11.8_cudnn8.7.0_0`), torchvision 0.15.2, spikingjelly
0.0.0.0.14, einops 0.6.1, cupy 12.0.0, and CUDA-related packages.

## Verdicts

A02: trained version AVAILABLE — public STEP Hugging Face checkpoint, HEAD 200.

A04: trained version AVAILABLE — public STEP Hugging Face checkpoint, HEAD 200.

A07: trained version AVAILABLE — public SpikingLM Hugging Face model.safetensors, HEAD 200.

A08: trained version needs training — simulator activation dumps are committed, but no trained checkpoint is provided.

B09: trained version AVAILABLE — LoAS sample checkpoints are committed locally.

C01: trained version AVAILABLE — five of six README-listed model/dataset rows have public checkpoint links; SMA-VGG/CIFAR10-DVS has no link.
