# GPU symmetry regression

Compare all 18 contractions, individually and together, against full CPU
contractions with an independently applied output-sector mask. The fixture uses
two unequal atoms and three periodic cells, real and complex tensors, both CPU
modes, and optionally DDLA/CUDA. It checks unrestricted, restricted and empty
output, as well as seeded signed, zero and fractional accumulation.

```sh
cmake -S unittests/gpu_symmetry -B build-symmetry \
  -DLIBCOMM_ROOT=/path/to/LibComm -DCEREAL_ROOT=/path/to/cereal \
  -DENABLE_GPU=ON -DDDLA_ROOT=/path/to/DDLA
cmake --build build-symmetry --parallel 2
ctest --test-dir build-symmetry --output-on-failure
```

The GPU test requires one GPU, one MPI rank, and LibDDLA 0.0.6 or newer built for CUDA. The DDLA package is discovered in
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
unrestricted behavior. The numerical regression also disables symmetry on a copied selector after each
filtered run to verify independent configuration of copied LRI objects.

## Select filters before contraction loops

`LRI::filter_atom` is an optional callable taking `LRI_Cal_Mode` and returning
a `const Filter_Atom*`. Each backend invokes it once before its OpenMP/atom loops;
all inner callbacks go directly to that pointer. An empty callable disables
filtering. The returned pointer must remain valid for the contraction. Custom
policies can use the same interface without changes to LRI.

`Filter_Atom_Symmetry_Wrapper` is a callable rather than a filter subclass. It
constructs CPU and (in GPU builds) GPU policies together and selects the existing
policy on each contraction. Its shared immutable configuration owns copies of
period and sector, keeping filter references valid after copies or temporary
caller data. Call `set_symmetry` again to change that configuration.

The regression asserts one selector invocation per contraction and stable filter
identity across repeated calls, checks backend switching, copied selectors,
caller configuration lifetime, and disabling symmetry. All 18 label expectations
continue to use the independent output mask described above.
