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

The GPU test requires one GPU, one MPI rank, and DDLA built for CUDA. Omit
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
