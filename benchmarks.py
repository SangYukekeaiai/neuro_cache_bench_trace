"""Model and dataset loaders for the trace-capture benchmarks.

The upstream repositories remain the source of the model definitions.  This
module only supplies small adapters for their input conventions and loads
their published weights with PyTorch's restricted unpickler.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import pathlib
import sys
import types
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
UPSTREAM = PROJECT_ROOT / "upstream"
DATA_ROOT = PROJECT_ROOT / "data"

SYNTHETIC_T = 4
SYNTHETIC_DATASET_SIZE = 64
SYNTHETIC_DATASET_SEED = 20261028
SYNTHETIC_WEIGHT_SEED = 1729

# A loader can leave a note here when an optional event dataset is absent.
BLOCKED_WORKLOADS: list[str] = []


class BenchmarkBlocked(RuntimeError):
    """A workload is valid but its external dataset is unavailable locally."""


class BinarySpikeDataset(Dataset[torch.Tensor]):
    """Deterministic binary input spikes with shape ``[T, C, H, W]``."""

    trace_split_name = "synthetic/test"

    def __init__(self, size: int = SYNTHETIC_DATASET_SIZE, timesteps: int = SYNTHETIC_T):
        generator = torch.Generator(device="cpu").manual_seed(SYNTHETIC_DATASET_SEED)
        self.spikes = torch.randint(
            0,
            2,
            (size, timesteps, 1, 8, 8),
            generator=generator,
            dtype=torch.uint8,
        ).to(torch.float32)
        self.labels = torch.arange(size, dtype=torch.long) % 10

    def __len__(self) -> int:
        return self.spikes.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.spikes[index]


class PreparedDataset(Dataset):
    """Apply an input adapter while retaining labels and split provenance."""

    def __init__(
        self,
        base: Dataset,
        converter: Callable[[Any], Any],
        split_name: str,
        labels: Any,
    ) -> None:
        self.base = base
        self.converter = converter
        self.trace_split_name = split_name
        self.labels = labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Any:
        return self.converter(self.base[index])


class SplitDataset(Dataset):
    """Prefer a test/validation split and fall back to train when necessary."""

    def __init__(
        self,
        preferred: PreparedDataset,
        fallback: PreparedDataset | None = None,
    ) -> None:
        self.preferred = preferred
        self.fallback = fallback
        self.trace_split_name = preferred.trace_split_name
        self.labels = preferred.labels

    def __len__(self) -> int:
        return len(self.preferred)

    def __getitem__(self, index: int) -> Any:
        return self.preferred[index]

    def choose(self, sample_count: int) -> PreparedDataset:
        if len(self.preferred) >= sample_count:
            return self.preferred
        if self.fallback is not None and len(self.fallback) >= sample_count:
            return self.fallback
        raise ValueError(
            f"neither {self.preferred.trace_split_name} nor "
            f"{getattr(self.fallback, 'trace_split_name', 'train')} has "
            f"{sample_count} samples"
        )


class SpikingGEMM(nn.Module):
    """A linear/GEMM block whose output is converted back to binary spikes."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"SpikingGEMM expects [batch, M, K], got {tuple(x.shape)}")
        return (self.linear(x) >= 0).to(dtype=x.dtype)


class SpikingFC(SpikingGEMM):
    """The final spiking fully connected layer (called with ``M == 1``)."""


class SyntheticSpikingNet(nn.Module):
    """Three binary-spike convolutions, an eight-token GEMM, and a final FC."""

    timesteps = SYNTHETIC_T

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.spiking_gemm = SpikingGEMM(32, 32)
        self.final_fc = SpikingFC(32, 10)

    @staticmethod
    def _spike(x: torch.Tensor) -> torch.Tensor:
        return (x >= 0).to(dtype=x.dtype)

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        if spikes.ndim != 5:
            raise ValueError(
                f"SyntheticSpikingNet expects [batch, T, C, H, W], got {tuple(spikes.shape)}"
            )
        if spikes.shape[1] != self.timesteps:
            raise ValueError(
                f"SyntheticSpikingNet expects T={self.timesteps}, got T={spikes.shape[1]}"
            )

        outputs = []
        weight_dtype = self.conv1.weight.dtype
        for timestep in range(self.timesteps):
            x = spikes[:, timestep].to(dtype=weight_dtype)
            x = self._spike(self.conv1(x))
            x = self._spike(self.conv2(x))
            x = self._spike(self.conv3(x))
            tokens = x.flatten(start_dim=2).transpose(1, 2)[:, :8, :]
            x = self.spiking_gemm(tokens)
            x = self.final_fc(x[:, :1, :])
            outputs.append(x)

        return torch.stack(outputs, dim=1)


def _ensure_python312_compat() -> None:
    """Brain-Cog's optional ``dv`` import still references removed ``imp``."""

    if sys.version_info < (3, 12) or "imp" in sys.modules:
        return
    imp_module = types.ModuleType("imp")
    imp_module.find_module = lambda name: importlib.util.find_spec(name)
    sys.modules["imp"] = imp_module


def _cifar10(split: str, split_name: str) -> PreparedDataset:
    return _cifar("cifar10", "label", (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), split, split_name)


def _cifar100(split: str, split_name: str) -> PreparedDataset:
    return _cifar("cifar100", "fine_label", (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), split, split_name)


def _cifar(
    name: str, label_key: str, mean: tuple[float, ...], std: tuple[float, ...], split: str, split_name: str
) -> PreparedDataset:
    from datasets import load_dataset

    base = load_dataset(name, cache_dir=str(DATA_ROOT / "hf_cifar"))["train" if split == "train" else "test"]

    def convert(item: Mapping[str, Any]) -> torch.Tensor:
        image = torch.from_numpy(np.asarray(item["img"], dtype=np.uint8).copy())
        image = image.permute(2, 0, 1).to(torch.float32).div_(255.0)
        return (image - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]

    labels = torch.as_tensor(base[label_key], dtype=torch.long)
    return PreparedDataset(base, convert, split_name, labels)


def _load_step_checkpoint(path: pathlib.Path) -> dict[str, Any]:
    # The checkpoint's args field contains argparse.Namespace; no executable
    # globals are needed to recover the state dict.
    with torch.serialization.safe_globals([argparse.Namespace]):
        return torch.load(path, map_location="cpu", weights_only=True)


def _load_a02() -> tuple[nn.Module, Dataset, int, str, str]:
    _ensure_python312_compat()
    sys.path.insert(0, str(UPSTREAM / "STEP" / "cls"))
    from models.static.spikformer_cifar import Spikformer

    checkpoint = _load_step_checkpoint(PROJECT_ROOT / "ckpts/A02/cifar100/model_best.pth.tar")
    model = Spikformer(
        step=4,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=100,
        embed_dim=384,
        num_heads=12,
        mlp_ratio=4,
        attn_scale=0.125,
        depths=4,
        tau=2.0,
        threshold=1.0,
        alpha=4.0,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    dataset = SplitDataset(_cifar100("test", "cifar100/test"), _cifar100("train", "cifar100/train"))
    return model, dataset, 4, "spikformer_4_384_cifar100", "cifar100"


def _load_a04() -> tuple[nn.Module, Dataset, int, str, str]:
    _ensure_python312_compat()
    sys.path.insert(0, str(UPSTREAM / "STEP" / "cls"))
    from models.static.sdt_cifar import SDTV1

    checkpoint = _load_step_checkpoint(PROJECT_ROOT / "ckpts/A04/cifar100/model_best.pth.tar")
    model = SDTV1(
        step=4,
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=100,
        embed_dim=512,
        num_heads=8,
        mlp_ratio=4,
        scale=0.125,
        depths=2,
        tau=2.0,
        threshold=1.0,
        alpha=4.0,
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    dataset = SplitDataset(_cifar100("test", "cifar100/test"), _cifar100("train", "cifar100/train"))
    return model, dataset, 4, "sdt_2_512_cifar100", "cifar100"


class _LoASVGGWrapper(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int):
        super().__init__()
        self.core = model
        self.timesteps = timesteps
        self.core.total_timestep = timesteps

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _prepare_loas_aux(self.core, images.device)
        logits = self.core(images, 0)
        return logits.mean(0)


class _LoASResNetWrapper(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int = 4):
        super().__init__()
        self.core = model
        self.timesteps = timesteps
        self.core.total_timestep = timesteps

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _prepare_loas_aux(self.core, images.device)
        logits = self.core((images, 0))
        return logits.mean(0)


def _prepare_loas_aux(model: nn.Module, device: torch.device) -> None:
    """Move/reset non-parameter firing-rate counters saved in LoAS objects."""

    for submodule in model.modules():
        counter = getattr(submodule, "spikerate", None)
        if isinstance(counter, torch.Tensor):
            submodule.spikerate = torch.zeros((), device=device, dtype=counter.dtype)


def _load_loas_model(path: pathlib.Path) -> nn.Module:
    sys.path.insert(0, str(UPSTREAM / "LoAS"))
    from archs.cifarsvhn import resnet, vgg
    from spikingjelly.clock_driven import neuron, surrogate
    from utils_for_snn_lth import computing_firerate

    from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Linear, MaxPool2d, Sequential

    safe = [
        vgg.VGG,
        resnet.ResNet,
        resnet.BasicBlock,
        computing_firerate,
        neuron.LIFNode,
        surrogate.ATan,
        Conv2d,
        Linear,
        BatchNorm2d,
        Sequential,
        MaxPool2d,
        AdaptiveAvgPool2d,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
    ]
    with torch.serialization.safe_globals(safe):
        model = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(model, nn.Module):
        raise TypeError(f"LoAS checkpoint did not contain a model: {type(model)}")
    # The committed objects predate two attributes added by the currently
    # installed SpikingJelly/LoAS runtime; restore their inference defaults.
    for submodule in model.modules():
        if isinstance(submodule, neuron.LIFNode) and not hasattr(submodule, "cupy_fp32_inference"):
            submodule.cupy_fp32_inference = False
        if isinstance(submodule, resnet.BasicBlock) and not hasattr(submodule, "total_timestep"):
            submodule.total_timestep = int(getattr(model, "total_timestep", 4))
    model.eval()
    return model


def _load_a08() -> tuple[nn.Module, Dataset, int, str, str]:
    model = _LoASVGGWrapper(
        _load_loas_model(PROJECT_ROOT / "ckpts/B09/vgg16_final_dict.pth.tar"), 32
    )
    dataset = SplitDataset(_cifar10("test", "cifar10/test"), _cifar10("train", "cifar10/train"))
    # vgg16_config() in Prosperity/simulator/configs.py declares time_steps=32.
    return model, dataset, 32, "vgg16_cifar10_loas", "cifar10"


def _load_b09_vgg() -> tuple[nn.Module, Dataset, int, str, str]:
    model = _LoASVGGWrapper(
        _load_loas_model(PROJECT_ROOT / "ckpts/B09/vgg16_final_dict.pth.tar"), 4
    )
    dataset = SplitDataset(_cifar10("test", "cifar10/test"), _cifar10("train", "cifar10/train"))
    return model, dataset, 4, "loas_vgg16", "cifar10"


def _load_b09_resnet() -> tuple[nn.Module, Dataset, int, str, str]:
    model = _LoASResNetWrapper(
        _load_loas_model(PROJECT_ROOT / "ckpts/B09/resnet19_final_dict.pth.tar"), 4
    )
    dataset = SplitDataset(_cifar10("test", "cifar10/test"), _cifar10("train", "cifar10/train"))
    return model, dataset, 4, "loas_resnet19", "cifar10"


class _SpikingLMWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.core = model

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.core(**batch).logits


class _SpikingLMInputDataset(Dataset):
    def __init__(self, raw: Any, tokenizer: Any, split_name: str):
        self.raw = raw
        self.tokenizer = tokenizer
        self.trace_split_name = split_name
        self.labels = torch.as_tensor(raw["label"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.raw[index]["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}


def _load_a07() -> tuple[nn.Module, Dataset, int, str, str]:
    from datasets import load_dataset
    from safetensors.torch import load_file
    from transformers import BertConfig, BertTokenizerFast

    from spikingjelly.activation_based import neuron

    class TorchLIFNode(neuron.LIFNode):
        def __init__(self, *args: Any, **kwargs: Any):
            kwargs["backend"] = "torch"
            super().__init__(*args, **kwargs)

    neuron.LIFNode = TorchLIFNode
    sys.path.insert(0, str(UPSTREAM / "SpikingLM"))
    from spiking_bert.modeling_spiking_bert import BertForMaskedLM

    checkpoint_dir = PROJECT_ROOT / "ckpts/A07"
    config = BertConfig.from_json_file(checkpoint_dir / "config.json")
    config.T = 4
    config._attn_implementation = "eager"
    model = BertForMaskedLM(config)
    state = load_file(str(checkpoint_dir / "model.safetensors"), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or set(missing) - {"cls.predictions.decoder.weight", "cls.predictions.decoder.bias"}:
        raise RuntimeError(
            f"unexpected SpikingLM checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.tie_weights()
    model.eval()
    # Keep future model construction in this process on the torch backend.
    neuron.LIFNode = TorchLIFNode

    raw = load_dataset("glue", "sst2", cache_dir=str(DATA_ROOT / "hf"))
    tokenizer = BertTokenizerFast.from_pretrained(str(checkpoint_dir), local_files_only=True)
    validation = _SpikingLMInputDataset(raw["validation"], tokenizer, "glue/sst2/validation")
    train = _SpikingLMInputDataset(raw["train"], tokenizer, "glue/sst2/train")
    dataset = SplitDataset(validation, train)
    return _SpikingLMWrapper(model), dataset, 4, "spikinglm_t4", "glue/sst2"


def _import_c01_module(filename: str, module_name: str) -> types.ModuleType:
    path = UPSTREAM / "SMA-AZO" / "SMA VGG" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_c01_state(model: nn.Module, path: pathlib.Path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def _c01_dataset(filename: str, split_name: str, frames: int, train: bool) -> PreparedDataset:
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

    if filename == "dvs128":
        base = DVS128Gesture(
            str(DATA_ROOT / "DVS128Gesture"),
            train=train,
            data_type="frame",
            split_by="number",
            frames_number=frames,
        )
    else:
        root = DATA_ROOT / "cifar10dvs" / ("train_dataset" if train else "test_dataset")
        base = CIFAR10DVS(str(root), data_type="frame", split_by="number", frames_number=frames)
    labels = torch.as_tensor([target for _path, target in base.samples], dtype=torch.long)
    return PreparedDataset(
        base,
        lambda item: torch.as_tensor(item[0], dtype=torch.float32),
        split_name,
        labels,
    )


class _C01Wrapper(nn.Module):
    def __init__(self, model: nn.Module, timesteps: int, module: types.ModuleType):
        super().__init__()
        self.core = model
        self.timesteps = timesteps
        self.module = module

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # The upstream model uses [T, B, C, H, W].
        self.module.train_flag = 0
        # Mean over T as upstream does: upstream/SMA-AZO/SMA VGG/vgg8_dvs128_SMA.py:406-408, :444-448.
        return self.core(frames.transpose(0, 1)).mean(0)


def _load_c01_dvs() -> tuple[nn.Module, Dataset, int, str, str]:
    module = _import_c01_module("vgg8_dvs128_SMA.py", "sma_azo_dvs128")
    model = module.VGG8()
    module.functional.set_step_mode(model, step_mode="m")
    del module.Net
    gc.collect()
    _load_c01_state(model, PROJECT_ROOT / "ckpts/C01/SMA-AZO-VGG_DVS128-Gesture.pth")
    model.eval()
    dataset = SplitDataset(
        _c01_dataset("dvs128", "dvs128_gesture/test", 16, False),
        _c01_dataset("dvs128", "dvs128_gesture/train", 16, True),
    )
    return _C01Wrapper(model, 16, module), dataset, 16, "sma_azo_vgg_dvs128_gesture", "dvs128_gesture"


def _load_c01_cifar() -> tuple[nn.Module, Dataset, int, str, str]:
    module = _import_c01_module("vgg8_cifar10dvs_SMA.py", "sma_azo_cifar10dvs")
    model = module.VGG8()
    module.functional.set_step_mode(model, step_mode="m")
    # The committed CIFAR10-DVS weights have four temporal attention channels;
    # the upstream class currently constructs two for T=10.
    for block in (model.block2[1], model.block3[1]):
        block.T_before_attention_conv = nn.Conv3d(10, 4, 1)
        block.T_after_attention_convs = nn.ModuleList([nn.Conv3d(4, 10, 1) for _ in range(4)])
    del module.Net
    gc.collect()
    _load_c01_state(model, PROJECT_ROOT / "ckpts/C01/SMA-AZO-VGG_CIFAR10-DVS.pth")
    model.eval()
    dataset = SplitDataset(
        _c01_dataset("cifar10dvs", "cifar10_dvs/test", 10, False),
        _c01_dataset("cifar10dvs", "cifar10_dvs/train", 10, True),
    )
    return _C01Wrapper(model, 10, module), dataset, 10, "sma_azo_vgg_cifar10_dvs", "cifar10_dvs"


def load_c01() -> list[tuple[nn.Module, Dataset, int, str, str]]:
    BLOCKED_WORKLOADS.clear()
    workloads: list[tuple[nn.Module, Dataset, int, str, str]] = []
    for label, loader in (("DVS128-Gesture", _load_c01_dvs), ("CIFAR10-DVS", _load_c01_cifar)):
        try:
            workloads.append(loader())
        except FileNotFoundError as exc:
            BLOCKED_WORKLOADS.append(f"{label}: {type(exc).__name__}: {exc}")
    if not workloads:
        raise BenchmarkBlocked("; ".join(BLOCKED_WORKLOADS))
    return workloads


def load_synthetic() -> tuple[nn.Module, Dataset[torch.Tensor], int, str, str]:
    """Return the deterministic synthetic model and binary-spike dataset."""

    dataset = BinarySpikeDataset()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SYNTHETIC_WEIGHT_SEED)
        model = SyntheticSpikingNet()
    model.eval()
    return model, dataset, SYNTHETIC_T, "synthetic_spiking", "synthetic_binary_spikes"


Loader = Callable[
    [], tuple[nn.Module, Dataset, int, str, str] | list[tuple[nn.Module, Dataset, int, str, str]]
]

REGISTRY: dict[str, Loader] = {
    "A02": _load_a02,
    "A04": _load_a04,
    "A07": _load_a07,
    "A08": _load_a08,
    "B09": lambda: [_load_b09_vgg(), _load_b09_resnet()],
    "C01": load_c01,
    "synthetic": load_synthetic,
}


def _evaluation_batch(dataset: Dataset, sample_count: int, seed: int) -> tuple[Dataset, list[int], Any]:
    selected = dataset.choose(sample_count) if hasattr(dataset, "choose") else dataset
    if sample_count > len(selected):
        raise ValueError(f"evaluation sample count {sample_count} exceeds dataset size {len(selected)}")
    indices = sorted(
        int(index)
        for index in np.random.default_rng(seed).choice(len(selected), sample_count, replace=False)
    )
    samples = [selected[index] for index in indices]
    first = samples[0]
    if isinstance(first, Mapping):
        batch = {
            key: torch.stack([sample[key] for sample in samples], dim=0)
            for key in first
        }
    else:
        batch = torch.stack([torch.as_tensor(sample) for sample in samples], dim=0)
    return selected, indices, batch


def _reset_spiking_state(model: nn.Module) -> None:
    try:
        from spikingjelly.activation_based import functional
    except ImportError:
        return
    functional.reset_net(model)


def _move_batch(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device)
    if isinstance(batch, Mapping):
        return {key: _move_batch(value, device) for key, value in batch.items()}
    raise TypeError(f"unsupported evaluation batch type {type(batch)}")


def _classification_accuracy(
    model: nn.Module, batch: Any, labels: torch.Tensor, device: torch.device
) -> tuple[int, int]:
    model.to(device=device).eval()
    with torch.inference_mode():
        output = model(_move_batch(batch, device))
    if not isinstance(output, torch.Tensor) or output.ndim != 2:
        raise ValueError(f"classifier returned expected [batch, classes], got {type(output)} {getattr(output, 'shape', None)}")
    predicted = output.argmax(dim=-1).to(device="cpu")
    expected = labels.to(device="cpu")
    return int((predicted == expected).sum()), int(expected.numel())


def _masked_language_batch(
    batch: Mapping[str, torch.Tensor], tokenizer: Any, seed: int, probability: float = 0.15
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    input_ids = batch["input_ids"].clone()
    labels = input_ids.clone()
    special = torch.tensor(
        [
            tokenizer.get_special_tokens_mask(row.tolist(), already_has_special_tokens=True)
            for row in input_ids
        ],
        dtype=torch.bool,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    probability_matrix = torch.full(input_ids.shape, probability, dtype=torch.float32)
    probability_matrix.masked_fill_(special, 0.0)
    masked = torch.bernoulli(probability_matrix, generator=generator).to(dtype=torch.bool)
    labels[~masked] = -100

    replaced = torch.bernoulli(
        torch.full(input_ids.shape, 0.8, dtype=torch.float32), generator=generator
    ).to(dtype=torch.bool) & masked
    masked_inputs = input_ids.clone()
    masked_inputs[replaced] = tokenizer.mask_token_id
    randomised = torch.bernoulli(
        torch.full(input_ids.shape, 0.5, dtype=torch.float32), generator=generator
    ).to(dtype=torch.bool) & masked & ~replaced
    random_words = torch.randint(
        len(tokenizer), input_ids.shape, generator=generator, dtype=input_ids.dtype
    )
    masked_inputs[randomised] = random_words[randomised]

    masked_batch = dict(batch)
    masked_batch["input_ids"] = masked_inputs
    return masked_batch, labels


def _masked_language_loss(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    device: torch.device,
) -> float:
    model.to(device=device).eval()
    model_batch = _move_batch(batch, device)
    model_batch["labels"] = labels.to(device=device)
    _reset_spiking_state(model)
    with torch.inference_mode():
        output = model(model_batch)
    if not isinstance(output, torch.Tensor) or output.ndim != 3:
        raise ValueError(
            f"masked-LM returned expected [batch, sequence, vocab], got "
            f"{type(output)} {getattr(output, 'shape', None)}"
        )
    mask = labels.ne(-100).to(device=device)
    return float(torch.nn.functional.cross_entropy(output[mask], labels.to(device=device)[mask]).item())


def _a07_random_baseline(model: nn.Module, device: torch.device, seed: int) -> nn.Module:
    core = model.core
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        random_core = type(core)(core.config)
    random_core.tie_weights()
    return _SpikingLMWrapper(random_core).to(device=device).eval()


def _evaluate_main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the seeded benchmark sanity rows.")
    parser.add_argument("--evaluate", action="store_true", required=True)
    parser.add_argument("--benchmark", action="append", choices=sorted(REGISTRY), dest="benchmarks")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")

    benchmark_ids = args.benchmarks or ["A02", "A04", "A07", "A08", "B09", "C01"]
    device = torch.device(args.device)
    for benchmark_id in benchmark_ids:
        BLOCKED_WORKLOADS.clear()
        try:
            loaded = REGISTRY[benchmark_id]()
        except BenchmarkBlocked as exc:
            print(f"{benchmark_id}: BLOCKED {exc}")
            continue
        workloads = [loaded] if isinstance(loaded, tuple) else loaded
        for model, dataset, _timesteps, arch_name, _dataset_name in workloads:
            selected, indices, batch = _evaluation_batch(dataset, args.n, args.seed)
            if benchmark_id == "A07":
                masked_batch, labels = _masked_language_batch(
                    batch, selected.tokenizer, args.seed
                )
                loss = _masked_language_loss(model, masked_batch, labels, device)
                random_model = _a07_random_baseline(model, device, args.seed + 1)
                random_loss = _masked_language_loss(random_model, masked_batch, labels, device)
                print(
                    f"{benchmark_id}/{arch_name} split={selected.trace_split_name} "
                    f"n={len(indices)} seed={args.seed} "
                    f"pseudo_loss={loss:.6f} random_init={random_loss:.6f}"
                )
                del random_model
            else:
                correct, count = _classification_accuracy(model, batch, selected.labels[indices], device)
                print(
                    f"{benchmark_id}/{arch_name} split={selected.trace_split_name} "
                    f"n={count} seed={args.seed} accuracy={correct}/{count}={correct / count:.6f}"
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for blocked in BLOCKED_WORKLOADS:
            print(f"{benchmark_id}: BLOCKED {blocked}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_evaluate_main())
