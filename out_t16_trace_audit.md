# Current Delta n=500 trace audit

Each directory was checked on the CPU with:

```bash
WCACHE_SRC=/u/yyu9/projects/neuro_cache_wcache/src \
  .venv/bin/python validate_trace.py <trace-directory>
```

| benchmark | trace directory | capture `captured_at` (UTC) | `validate_trace.py` | loader compatibility | verdict | note |
|---|---|---|---|---|---|---|
| A02 | `A02/spikformer_4_384_cifar100_T4_n500` | 2026-11-10 02:19:52 | PASS (exit 0) | Yes | **valid** | Recaptured after the CIFAR-100 loader/checkpoint fix; all 12 layers passed. |
| A04 | `A04/sdt_2_512_cifar100_T4_n500` | 2026-11-10 03:54:16 | PASS (exit 0) | Yes | **valid** | Recaptured after the CIFAR-100 loader/checkpoint fix; all 16 layers passed. |
| A07 | `A07/spikinglm_t4_T4_n500` | 2026-11-07 13:34:01 | PASS (exit 0) | Yes | **valid (structural only)** | `chunk_size=25`; all 72 layers passed. The n=500 pass is structural, not numerical-equivalence evidence; no unchunked n=500 reference exists because of OOM. |
| A08 | `A08/vgg16_cifar10_loas_T32_n500` | 2026-11-07 12:59:56 | PASS (exit 0) | Yes | **valid** | Loader is unchanged; all 13 layers passed. |
| B09 | `B09/loas_resnet19_T4_n500` | 2026-11-07 13:01:49 | PASS (exit 0; 6 expected WARNs) | Yes | **valid** | WARNs are the validator's expected replay exclusions for stride-2 layers. |
| B09 | `B09/loas_vgg16_T4_n500` | 2026-11-07 13:01:46 | PASS (exit 0) | Yes | **valid** | Loader is unchanged; all 13 layers passed. |
| C01 | `C01/sma_azo_vgg_cifar10_dvs_T10_n500` | 2026-11-07 13:40:09 | PASS (exit 0) | Yes | **valid (structural only)** | `chunk_size=25`; all 6 layers passed. The n=500 pass is structural, not numerical-equivalence evidence; no unchunked n=500 reference exists because of OOM. |
| C01 | `C01/sma_azo_vgg_dvs128_gesture_T16_n500` | 2026-11-07 13:39:55 | PASS (exit 0) | Yes | **valid (structural only)** | `chunk_size=25`; all 6 layers passed. The n=500 pass is structural, not numerical-equivalence evidence; no unchunked n=500 reference exists because of OOM. |

The campaign command passed `--seed 0`. The explicit expected-index artifact is
`out_t13_cpu_selection/selection_seed0_n500.json`; it records the sorted
`numpy.random.default_rng(0)` selections used for the CPU check and matches the
current A02/A04 selections. The Nov 7 manifests do not independently record the
GPU capture seed, so that remains an open provenance gap rather than a fact
closed by inference from matching indices.
