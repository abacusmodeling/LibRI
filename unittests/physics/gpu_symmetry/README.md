# GPU symmetry regression

Compare all 18 contractions, individually and together, against full CPU
contractions with an independently applied output-sector mask. The fixture uses
two unequal atoms and three periodic cells, real and complex tensors, both CPU
modes, and optionally DDLA/CUDA. It checks unrestricted, restricted and empty
output, as well as seeded signed, zero and fractional accumulation.

```sh
cmake -S unittests/physics/gpu_symmetry -B build-symmetry \
  -DLIBCOMM_ROOT=/path/to/LibComm -DCEREAL_ROOT=/path/to/cereal \
  -DENABLE_GPU=ON -DDDLA_ROOT=/path/to/DDLA
cmake --build build-symmetry --parallel 2
ctest --test-dir build-symmetry --output-on-failure
```

The GPU test requires one GPU, one MPI rank, and a CUDA-enabled LibDDLA
installation exposing the handle-selected public computational API
(`gemmVbatched<T>` and `gemmVbatched2s<T>`, without a backend template argument).
The version number alone is insufficient: older 0.0.6 installations expose a
different API. The consumer defines only `__GPU_RI` and `__DDLA_RI`; GPU
capability is read from DDLA's generated configuration header. The DDLA package is discovered in
either `lib/cmake/LibDDLA` or `lib64/cmake/LibDDLA`; only its public headers
and imported CMake target are used. Omit
`ENABLE_GPU` and `DDLA_ROOT` to check a CPU-only build. The library headers can
be selected independently with `LIBRI_ROOT`, allowing the same test to reproduce
failures in an earlier revision.

For OpenMP coverage or an isolated failing contraction:

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  mpirun -np 1 build-symmetry/test_lri_gpu_symmetry
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  mpirun -np 1 build-symmetry/test_lri_gpu_symmetry 1 gpu real
```

Arguments select the contraction index (`0`–`17`, `18` for the combined sum,
or `-1` for all), backend (`cpu`, `gpu`, `all`) and scalar (`real`, `complex`,
`all`). Index 1 is `a1b2_a2b1`: filtering `(Aa01,Aa2)` while summing over `Aa2`
incorrectly removes contributions to the accepted output pair `(Aa01,Ab01)`.
The off-diagonal sector with a nonzero translation detects this error.

## GPU symmetry callback contract

Arguments follow the enclosing loops from outermost to innermost. `filter_for31`
always returns false (intermediate construction); `filter_for32` rejects only
output pairs outside the sector. **Bold argument tuples changed in this refactor.**

| Label | Family | `filter_for31` arguments | `filter_for32` arguments | Output pair screened by `filter_for32` |
| --- | --- | --- | --- | --- |
| `a0b0_a1b1` | `a01b01_a01b01` | `(Aa01, Ab01, Ab2)` | `(Aa01, Aa2, Ab2)` | `(Aa2, Ab2)` |
| `a0b1_a1b0` | `a01b01_a01b01` | `(Aa01, Ab01, Ab2)` | `(Aa01, Aa2, Ab2)` | `(Aa2, Ab2)` |
| `a0b0_a1b2` | `a01b01_a01b2` | **`(Ab01, Ab2, Aa01)`** | **`(Aa01, Ab01, Aa2)`** | `(Aa2, Ab01)` |
| `a0b1_a1b2` | `a01b01_a01b2` | **`(Ab01, Ab2, Aa01)`** | **`(Aa01, Ab01, Aa2)`** | `(Aa2, Ab01)` |
| `a0b2_a1b0` | `a01b01_a01b2` | **`(Ab01, Ab2, Aa01)`** | **`(Aa01, Ab01, Aa2)`** | `(Aa2, Ab01)` |
| `a0b2_a1b1` | `a01b01_a01b2` | **`(Ab01, Ab2, Aa01)`** | **`(Aa01, Ab01, Aa2)`** | `(Aa2, Ab01)` |
| `a0b0_a2b1` | `a01b01_a2b01` | **`(Aa01, Aa2, Ab01)`** | `(Aa01, Ab01, Ab2)` | `(Aa01, Ab2)` |
| `a0b1_a2b0` | `a01b01_a2b01` | **`(Aa01, Aa2, Ab01)`** | `(Aa01, Ab01, Ab2)` | `(Aa01, Ab2)` |
| `a1b0_a2b1` | `a01b01_a2b01` | **`(Aa01, Aa2, Ab01)`** | `(Aa01, Ab01, Ab2)` | `(Aa01, Ab2)` |
| `a1b1_a2b0` | `a01b01_a2b01` | **`(Aa01, Aa2, Ab01)`** | `(Aa01, Ab01, Ab2)` | `(Aa01, Ab2)` |
| `a0b0_a2b2` | `a01b01_a2b2` | `(Aa01, Aa2, Ab2)` | **`(Aa01, Ab01, Ab2)`** | `(Aa01, Ab01)` |
| `a0b1_a2b2` | `a01b01_a2b2` | `(Aa01, Aa2, Ab2)` | **`(Aa01, Ab01, Ab2)`** | `(Aa01, Ab01)` |
| `a1b0_a2b2` | `a01b01_a2b2` | `(Aa01, Aa2, Ab2)` | **`(Aa01, Ab01, Ab2)`** | `(Aa01, Ab01)` |
| `a1b1_a2b2` | `a01b01_a2b2` | `(Aa01, Aa2, Ab2)` | **`(Aa01, Ab01, Ab2)`** | `(Aa01, Ab01)` |
| `a1b2_a2b1` | `a01b2_a2b01` | **`(Ab01, Ab2, Aa01)`** | **`(Aa01, Aa2, Ab01)`** | `(Aa01, Ab01)` |
| `a0b2_a2b0` | `a01b2_a2b01` | **`(Ab01, Ab2, Aa01)`** | **`(Aa01, Aa2, Ab01)`** | `(Aa01, Ab01)` |
| `a0b2_a2b1` | `a01b2_a2b01` | **`(Ab01, Ab2, Aa01)`** | **`(Aa01, Aa2, Ab01)`** | `(Aa01, Ab01)` |
| `a1b2_a2b0` | `a01b2_a2b01` | **`(Ab01, Ab2, Aa01)`** | **`(Aa01, Aa2, Ab01)`** | `(Aa01, Ab01)` |

`filter_for1(Aa01)` is a no-op for the first two families (Aa01 is intermediate),
and checks left-output membership for the other three. `filter_for1(Ab01)` checks
right-output membership wherever it occurs. All `filter_for2` calls are no-ops
except the final `(Aa01, Ab01)` stage of `a01b2_a2b01`, which checks that output
pair. In that family's second stage, the former inner
`filter_for2(Aa01, Ab01)` check is absorbed into
**`filter_for32(Aa01, Aa2, Ab01)`**; `Aa2` remains an unrestricted summation atom.

For two cell-bearing output atoms the relative cell is `(Rb - Ra) % period`;
with `Aa01` at the origin it is `Rb % period`. An empty sector retains the existing
unrestricted behavior. The numerical regression also disables symmetry after each
filtered run to verify replacement of the previous backend's filter.

## GPU staged callback behavior

The first digit in `filter_GPU_forXY` is the loop depth, the second is the
occurrence/stage. The loop call sites and their argument order are unchanged.

- `F`: return false (do not screen at this position).
- `L`: reject atoms absent from the sector's left/output atom identities (`Aa01`).
- `R`: reject atoms absent from the sector's right/output atom identities (`Ab01`).
- `P`: reject only if the final output pair/cell is outside the sector.
- `—`: no such call for this label. **Bold** entries gain screening in this change.

| Label | 11 | 12 | 13 | 21 | 22 | 23 | 31 | 32 | Final output pair |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `a0b0_a1b1` | F | F | — | F | F | — | F | P | `(Aa2, Ab2)` |
| `a0b1_a1b0` | F | F | — | F | F | — | F | P | `(Aa2, Ab2)` |
| `a0b0_a1b2` | R | F | — | F | F | — | F | P | `(Aa2, Ab01)` |
| `a0b1_a1b2` | R | F | — | F | F | — | F | P | `(Aa2, Ab01)` |
| `a0b2_a1b0` | R | F | — | F | F | — | F | P | `(Aa2, Ab01)` |
| `a0b2_a1b1` | R | F | — | F | F | — | F | P | `(Aa2, Ab01)` |
| `a0b0_a2b1` | L | **L** | — | F | F | — | F | P | `(Aa01, Ab2)` |
| `a0b1_a2b0` | L | **L** | — | F | F | — | F | P | `(Aa01, Ab2)` |
| `a1b0_a2b1` | L | **L** | — | F | F | — | F | P | `(Aa01, Ab2)` |
| `a1b1_a2b0` | L | **L** | — | F | F | — | F | P | `(Aa01, Ab2)` |
| `a0b0_a2b2` | L | **L** | — | F | **P** | — | F | P | `(Aa01, Ab01)` |
| `a0b1_a2b2` | L | **L** | — | F | **P** | — | F | P | `(Aa01, Ab01)` |
| `a1b0_a2b2` | L | **L** | — | F | **P** | — | F | P | `(Aa01, Ab01)` |
| `a1b1_a2b2` | L | **L** | — | F | **P** | — | F | P | `(Aa01, Ab01)` |
| `a1b2_a2b1` | R | **L** | **L** | F | F | **P** | F | P | `(Aa01, Ab01)` |
| `a0b2_a2b0` | R | **L** | **L** | F | F | **P** | F | P | `(Aa01, Ab01)` |
| `a0b2_a2b1` | R | **L** | **L** | F | F | **P** | F | P | `(Aa01, Ab01)` |
| `a1b2_a2b0` | R | **L** | **L** | F | F | **P** | F | P | `(Aa01, Ab01)` |

`for21` and `for31` remain no-ops. In particular, `for22(Aa01,Aa2)` in
`a01b2_a2b01` must not screen its intermediate `Aa2`; its final pair is checked
by `for32(Aa01,Aa2,Ab01)` and `for23(Aa01,Ab01)`. By contrast,
`for22(Aa01,Ab01)` in `a01b01_a2b2` already has both output atoms and can screen.

The `for32` arguments and selected pair are:

| Family | Arguments | Pair screened |
| --- | --- | --- |
| `a01b01_a01b01` | `Aa01, Aa2, Ab2` | `Aa2, Ab2` |
| `a01b01_a01b2` | `Aa01, Ab01, Aa2` | `Aa2, Ab01` |
| `a01b01_a2b01` | `Aa01, Ab01, Ab2` | `Aa01, Ab2` |
| `a01b01_a2b2` | `Aa01, Ab01, Ab2` | `Aa01, Ab01` |
| `a01b2_a2b01` | `Aa01, Aa2, Ab01` | `Aa01, Ab01` |

For cell-bearing left outputs, use the relative translation `(Rb-Ra) % period`;
for origin-based `Aa01`, use `Rb % period`. Empty sector means unrestricted
output, matching the existing API.

The CPU-host `gpu_symmetry_filters` test invokes all GPU callback positions
through `Filter_Atom`, checking 31,104 label/sector/atom/cell combinations against
independent output-index and sector membership expectations. It needs no DDLA or
GPU. It checks intermediate cells that are outside the sector while the final
output is inside, off-diagonal pairs, periodic translations and empty output.
