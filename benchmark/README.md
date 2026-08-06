# Benchmark guide

Run every command from the repository root. Each entrypoint accepts case-selection flags; train and standalone VRAM benchmarks also accept a batch override.

```bash
python benchmark/train_tied.py --select kangaroo --batch 4
python benchmark/vram_precomputed.py --select 1,14-18 --batch 16
python benchmark/accuracy_tied.py --list
```

## Methodology

**Train benchmarks.** ORDA entry points are `torch._dynamo.disable`d, so a compiled caller sees a graph break at the loss. Train benchmarks therefore measure a full training pipeline where the baseline wraps everything in `torch.compile`, while the ORDA run compiles the model only. Each backend runs in the configuration that suits it.

**VRAM benchmarks.** These measure peak VRAM of the loss section alone. ORDA runs with a fixed `num_chunks=32` — the project's default recommended maximum — so the reported figure reflects the reduction available at that chunk count.

## Flags

All eight routers support the following flags.

| Flag | Meaning |
| --- | --- |
| `--list` | Print the supported case IDs and exit. Cannot be combined with a selection flag. |
| `--select ID_OR_NUMBER [...]` | Run case IDs, one-based numbers, or ranges, for example `kangaroo 1,3,5-7`. May be repeated. |
| `--all` | Run every case. Cannot be combined with another selection flag. |

`--batch N` is supported only by the train and standalone VRAM routers. `N` must be a positive integer and overrides the selected profile batch for that invocation. Accuracy has fixed `(sequence length, batch)` shapes and does not accept `--batch`.

## Profiles and model configuration

Edit [`_lib/config/profiles.py`](_lib/config/profiles.py) to change benchmark sizing.

| What to change | Location |
| --- | --- |
| Train teacher/student model dimensions | `SMALL_PROFILE` and `LARGE_PROFILE`: `teacher_config`, `student_config` |
| Train vocabulary, sequence lengths, default batch, warmup, measured steps and passes | `SMALL_PROFILE` and `LARGE_PROFILE` |
| Standalone VRAM vocabulary, sequence lengths, dimensions, default batch and sweep iterations | `VRAM_SMALL` and `VRAM_LARGE` |
| Device threshold that selects small vs. large | `SMALL_PROFILE_VRAM_GB` |
| Accuracy's fixed vocabulary, shape/batch pairs and temperature | `accuracy_profile()` |

Small is selected for a device with less than 16 GB VRAM, or when VRAM is unknown; otherwise large is selected. The default batch is 8 in every train and standalone VRAM profile.

The three tied special workloads are defined in [`_lib/suites/specs.py`](_lib/suites/specs.py) as `SPECIAL_CASES`: `kangaroo`, `recursive_transformer`, and `r_drop`.

## Routers and supported cases

| Router | Family | Case set |
| --- | --- | --- |
| `accuracy_tied.py` | tied | 30 generic tied modes plus `kangaroo`, `recursive_transformer`, `r_drop` |
| `accuracy_separate.py` | separate | 30 generic separate modes |
| `accuracy_precomputed.py` | precomputed | 13 hidden-cache divergences, two aligned-span cases, and seven precomputed-input cases |
| `train_tied.py` | tied | `kangaroo`, `recursive_transformer`, `r_drop` |
| `train_separate.py` | separate | 30 generic separate modes |
| `train_precomputed.py` | precomputed | 13 divergences, two aligned-span cases, cross-vocab projection, and two multi-teacher cases |
| `vram_separate.py` | separate | 30 generic separate modes |
| `vram_precomputed.py` | precomputed | The same 18 cases as `train_precomputed.py` |

There is no `vram_tied.py`: the special tied workloads report VRAM as part of `train_tied.py`.

### Exact case IDs

#### `accuracy_tied.py`

```text
 1. tied-trained-forward
 2. tied-trained-jeffreys
 3. tied-trained-reverse
 4. tied-trained-alpha_divergence
 5. tied-trained-skew_forward
 6. tied-trained-skew_js
 7. tied-trained-entropy_adaptive
 8. tied-trained-skew_reverse
 9. tied-trained-renyi
10. tied-trained-renyi_reverse
11. tied-trained-dkd
12. tied-trained-abkd
13. tied-trained-triangular
14. tied-frozen-forward
15. tied-frozen-jeffreys
16. tied-frozen-reverse
17. tied-frozen-alpha_divergence
18. tied-frozen-skew_forward
19. tied-frozen-skew_js
20. tied-frozen-entropy_adaptive
21. tied-frozen-skew_reverse
22. tied-frozen-renyi
23. tied-frozen-renyi_reverse
24. tied-frozen-dkd
25. tied-frozen-abkd
26. tied-frozen-triangular
27. tied-cotrained-mutual
28. tied-cotrained-cojeffreys
29. tied-cotrained-teacher-ce-mutual
30. tied-cotrained-teacher-ce-cojeffreys
31. kangaroo
32. recursive_transformer
33. r_drop
```

#### `accuracy_separate.py`

```text
 1. separate-frozen-forward
 2. separate-frozen-jeffreys
 3. separate-frozen-reverse
 4. separate-frozen-alpha_divergence
 5. separate-frozen-skew_forward
 6. separate-frozen-skew_js
 7. separate-frozen-entropy_adaptive
 8. separate-frozen-skew_reverse
 9. separate-frozen-renyi
10. separate-frozen-renyi_reverse
11. separate-frozen-dkd
12. separate-frozen-abkd
13. separate-frozen-triangular
14. separate-trained-forward
15. separate-trained-jeffreys
16. separate-trained-reverse
17. separate-trained-alpha_divergence
18. separate-trained-skew_forward
19. separate-trained-skew_js
20. separate-trained-entropy_adaptive
21. separate-trained-skew_reverse
22. separate-trained-renyi
23. separate-trained-renyi_reverse
24. separate-trained-dkd
25. separate-trained-abkd
26. separate-trained-triangular
27. separate-cotrained-mutual
28. separate-cotrained-cojeffreys
29. separate-cotrained-teacher-ce-mutual
30. separate-cotrained-teacher-ce-cojeffreys
```

#### `accuracy_precomputed.py`

```text
 1. precomputed-hidden-forward
 2. precomputed-hidden-jeffreys
 3. precomputed-hidden-reverse
 4. precomputed-hidden-alpha_divergence
 5. precomputed-hidden-skew_forward
 6. precomputed-hidden-skew_js
 7. precomputed-hidden-entropy_adaptive
 8. precomputed-hidden-skew_reverse
 9. precomputed-hidden-renyi
10. precomputed-hidden-renyi_reverse
11. precomputed-hidden-dkd
12. precomputed-hidden-abkd
13. precomputed-hidden-triangular
14. aligned_span_binary_ce
15. aligned_span_binary_kl
16. cross_vocab_projection
17. cross_vocab_end_to_end_precomputed_loss
18. aligned_span_live_cross_vocab
19. precomputed_span_teacher_cache_build
20. multi_teacher_arithmetic
21. multi_teacher_geometric
22. multi_teacher_gjs
```

#### `train_tied.py`

```text
1. kangaroo
2. recursive_transformer
3. r_drop
```

#### `train_separate.py` and `vram_separate.py`

```text
 1. trained_abkd
 2. trained_alpha_divergence
 3. trained_dkd
 4. trained_entropy_adaptive
 5. trained_forward
 6. trained_jeffreys
 7. trained_renyi_reverse
 8. trained_renyi
 9. trained_reverse
10. trained_skew_forward
11. trained_skew_js
12. trained_skew_reverse
13. trained_triangular
14. frozen_abkd
15. frozen_alpha_divergence
16. frozen_dkd
17. frozen_entropy_adaptive
18. frozen_forward
19. frozen_jeffreys
20. frozen_renyi_reverse
21. frozen_renyi
22. frozen_reverse
23. frozen_skew_forward
24. frozen_skew_js
25. frozen_skew_reverse
26. frozen_triangular
27. cotrained_mutual
28. cotrained_cojeffreys
29. cotrained_teacher_ce_mutual
30. cotrained_teacher_ce_cojeffreys
```

#### `train_precomputed.py` and `vram_precomputed.py`

```text
 1. abkd
 2. alpha_divergence
 3. dkd
 4. entropy_adaptive
 5. forward
 6. jeffreys
 7. renyi_reverse
 8. renyi
 9. reverse
10. skew_forward
11. skew_js
12. skew_reverse
13. triangular
14. aligned_span_binary_ce
15. aligned_span_binary_kl
16. cross_vocab_end_to_end_precomputed_loss
17. multi_teacher_arithmetic
18. multi_teacher_geometric
```

Use `--list` as the source of truth when adding or selecting cases; it reflects the active router registry.
