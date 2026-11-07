import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from capture_traces import _capture, _capture_chunked, _capture_workload, _stack_layer, _stack_samples


class MixedBinarityCaptureTest(unittest.TestCase):
    def test_binary_then_non_binary_layer_is_excluded_without_partial_capture(self) -> None:
        class MixedInputModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.target = nn.Linear(2, 2)

            def forward(self, _samples: torch.Tensor) -> None:
                self.target(torch.tensor([[0.0, 1.0]]))
                self.target(torch.tensor([[0.0, 0.5]]))

        captures, excluded = _capture(
            MixedInputModel().eval(),
            torch.zeros(1),
            sample_count=1,
            timestep_count=2,
            device=torch.device("cpu"),
            allow_unfired=False,
        )

        self.assertEqual(captures, [])
        self.assertEqual(
            excluded,
            {"target": "weight-bearing input is not exactly binary; excluded from capture"},
        )

    def test_exclusion_does_not_reuse_a_layer_number(self) -> None:
        timestep_count, sample_count = 4, 2

        class ThreeConvModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.convA = nn.Conv2d(1, 1, 3, padding=1, bias=False)
                self.convMID = nn.Conv2d(1, 1, 3, padding=1, bias=False)
                self.convB = nn.Conv2d(1, 1, 3, padding=1, bias=False)

            def forward(self, samples: torch.Tensor) -> torch.Tensor:
                outputs = []
                for timestep in range(timestep_count):
                    current = samples[timestep * sample_count : (timestep + 1) * sample_count]
                    first = self.convA(current)
                    # The first call consumes layer 02, then the second call
                    # in this same timestep excludes it before the later
                    # survivor gets its first record.
                    middle = self.convMID(current)
                    middle = middle + self.convMID(current * 0.5)
                    later = self.convB(current)
                    outputs.append(first + middle + later)
                return torch.cat(outputs, dim=0)

        torch.manual_seed(0)
        samples = (torch.rand(timestep_count * sample_count, 1, 4, 4) > 0.5).float()
        captures, excluded = _capture(
            ThreeConvModel().eval(),
            samples,
            sample_count=sample_count,
            timestep_count=timestep_count,
            device=torch.device("cpu"),
            allow_unfired=False,
        )

        names = [name for name, _module, _values in captures]
        self.assertEqual(names, ["layer_01_convA", "layer_03_convB"])
        self.assertEqual(
            excluded,
            {"convMID": "weight-bearing input is not exactly binary; excluded from capture"},
        )
        self.assertEqual([len(values) for _name, _module, values in captures], [4, 4])
        for _name, _module, values in captures:
            self.assertEqual([value.shape for value in values], [(sample_count, 1, 4, 4)] * 4)
            self.assertTrue(
                all(np.all((value == 0) | (value == 1)) for value in values)
            )


class ChunkedCaptureTest(unittest.TestCase):
    def test_first_chunk_resets_state_from_an_earlier_forward(self) -> None:
        from spikingjelly.activation_based import neuron

        model = nn.Sequential(neuron.IFNode(v_threshold=1.0), nn.Linear(2, 1)).eval()
        dataset = [torch.full((2,), 0.6) for _ in range(4)]
        full, _ = _capture(copy.deepcopy(model), _stack_samples(dataset), 4, 1, torch.device("cpu"), False)
        with torch.no_grad():
            model(_stack_samples(dataset[:2]))
        chunked, _ = _capture_chunked(model, dataset, list(range(4)), 2, 1, torch.device("cpu"), False)
        self.assertEqual(len(chunked), len(full))
        for original, actual in zip(full, chunked):
            np.testing.assert_array_equal(_stack_layer(*actual, 4)[0], _stack_layer(*original, 4)[0])

    def test_chunk_metadata_for_all_benchmarks_and_unchunked_capture(self) -> None:
        dataset = [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])]
        with tempfile.TemporaryDirectory() as temp:
            for benchmark in ("A07", "C01", "synthetic"):
                for chunk in (None, 1):
                    with self.subTest(benchmark=benchmark, chunk=chunk):
                        args = SimpleNamespace(n=2, seed=0, chunk=chunk, allow_unfired=False,
                                               out=Path(temp), device="cpu", benchmark=benchmark)
                        output = _capture_workload(args, torch.device("cpu"),
                                                   nn.Sequential(nn.Linear(2, 1)), dataset,
                                                   1, "toy", "toy/test")
                        meta = json.loads((output / "meta.json").read_text())
                        if chunk is None:
                            self.assertNotIn("chunk_size", meta)
                            self.assertNotIn("numerical_note", meta)
                        else:
                            self.assertEqual(meta["chunk_size"], chunk)
                            self.assertIn("may differ", meta["numerical_note"])
                            if benchmark == "C01":
                                for fact in ("n=5, chunk=2 only", "2.9296875e-4", "layers 1-2 were exact",
                                             "4 digits", "DVS128-Gesture was bit-identical", "OOM", "structural"):
                                    self.assertIn(fact, meta["numerical_note"])

    def test_explicit_timesteps_with_single_sample_tail(self) -> None:
        from spikingjelly.activation_based import layer

        class MultiStepModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = layer.Conv2d(1, 1, 1, step_mode="m")
                self.fc = nn.Linear(4, 3)

            def forward(self, samples):
                steps = samples.unsqueeze(0).repeat(3, 1, 1, 1, 1)
                self.conv(steps)
                return self.fc(steps.flatten(2))

        dataset = [torch.full((1, 2, 2), float(index % 2)) for index in range(5)]
        model = MultiStepModel().eval()
        full, _ = _capture(model, _stack_samples(dataset), 5, 3, torch.device("cpu"), False)
        chunked, _ = _capture_chunked(model, dataset, list(range(5)), 2, 3, torch.device("cpu"), False)
        for original, actual in zip(full, chunked):
            self.assertEqual(actual[0], original[0])
            np.testing.assert_array_equal(_stack_layer(*actual, 5)[0], _stack_layer(*original, 5)[0])

    def test_state_sample_order_and_exclusions_match_full_batch(self) -> None:
        from spikingjelly.activation_based import neuron

        class StatefulModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.early = nn.Linear(4, 4)
                self.late = nn.Linear(4, 4)
                self.conv = nn.Conv2d(1, 1, 1)
                self.fc = nn.Linear(4, 3)
                self.lif = neuron.LIFNode(v_threshold=0.4)

            def forward(self, batch):
                x = batch["x"]
                for timestep in range(3):
                    self.early(x)
                    spikes = self.lif(torch.full_like(x, 0.6))
                    late_input = spikes.clone()
                    if timestep == 1:
                        late_input[x[:, 0] == 0.5] = 0.5
                    self.late(late_input)
                    self.conv(spikes.reshape(-1, 1, 2, 2))
                    self.fc(spikes)
                return spikes

        dataset = [{"x": torch.full((4,), value)} for value in (0., 1., 0.5, 1., 0.)]
        initial = StatefulModel().eval()
        full, excluded = _capture(
            copy.deepcopy(initial), _stack_samples(dataset), 5, 3, torch.device("cpu"), False,
        )
        self.assertEqual([name for name, _, _ in full], ["layer_02_conv", "layer_03_fc"])
        self.assertEqual(set(excluded), {"early", "late"})
        for chunk_size in (1, 2, 5):
            with self.subTest(chunk_size=chunk_size):
                chunked, chunk_excluded = _capture_chunked(
                    copy.deepcopy(initial), dataset, list(range(5)), chunk_size,
                    3, torch.device("cpu"), False,
                )
                self.assertEqual(chunk_excluded, excluded)
                self.assertEqual([name for name, _, _ in chunked], [name for name, _, _ in full])
                for original, actual in zip(full, chunked):
                    expected_array, expected_shape, expected_gemm = _stack_layer(*original, 5)
                    actual_array, actual_shape, actual_gemm = _stack_layer(*actual, 5)
                    np.testing.assert_array_equal(actual_array, expected_array)
                    self.assertEqual(actual_shape, expected_shape)
                    self.assertEqual(actual_gemm, expected_gemm)
                    self.assertEqual(actual_array.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
