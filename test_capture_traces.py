import unittest

import numpy as np
import torch
from torch import nn

from capture_traces import _capture


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


if __name__ == "__main__":
    unittest.main()
