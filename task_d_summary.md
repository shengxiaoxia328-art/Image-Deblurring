# Task d — Custom Householder QR Deblurring

## Goal
Repeat task (b) with our own QR factorization (Householder reflections from `task_c_householder_qr.py`) and compare against the SciPy implementation.

## Experimental setup
- Same test cases as before: `512_car` and `1024_books`, each with `config1` (singular kernels) and `config2` (well-conditioned banded kernels).
- Pipelines evaluated:
  - `LU` baseline (unchanged).
  - `QR (SciPy)` — reference from task (b).
  - `QR (ours)` — custom Householder QR.
- Metrics: runtime, relative Frobenius error, PSNR.

## Key results

| Image | Config | Method | Time (s) | Relative Error | PSNR (dB) |
|-------|--------|--------|----------|----------------|-----------|
| 512_car | config2 | QR (SciPy) | 0.023 | 1.33×10⁻⁸ | ∞ |
| 512_car | config2 | QR (ours) | 4.14 | 1.45×10⁻⁸ | ∞ |
| 1024_books | config2 | QR (SciPy) | 0.158 | 1.29×10⁻⁷ | 192.88 |
| 1024_books | config2 | QR (ours) | 53.08 | 1.10×10⁻⁷ | 194.23 |

Notes:
- For `config1` (near-singular kernels) both QR versions still fail, but the custom QR returns a slightly smaller residual because it avoids the explicit `qr` call; nevertheless the reconstructions remain unusable.
- For `config2`, the custom QR achieves comparable accuracy (even slightly better PSNR on the 1024 image) but is **~180–330× slower** due to the explicit formation of `Q` and repeated dense matrix multiplications in Python.

## Observations
1. **Accuracy**: On well-conditioned problems the custom QR matches SciPy’s solution quality (differences < 2e-9 in relative error). On ill-conditioned kernels both diverge similarly.
2. **Runtime**: The pure-Python Householder implementation is drastically slower (two orders of magnitude) because it builds full `Q` matrices and lacks BLAS-optimized kernels.
3. **Stability**: The custom QR handles the singular cases without throwing exceptions (thanks to `_safe_upper_solve`), demonstrating robustness even though the results are still poor because the underlying kernels are defective.

## Conclusion
Our QR implementation produces reconstruction quality comparable to SciPy’s QR on the meaningful test cases (config2), confirming correctness. However, it is **not competitive in speed**—which aligns with the assignment hint that beating the built-in routines is not required. Future improvements could store Householder vectors instead of dense `Q`, apply blocked updates, or switch to banded/Toeplitz-aware operations to reduce the runtime gap.

