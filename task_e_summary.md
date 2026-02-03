# Task e — Least-Squares Improvements and Kernel Diagnosis

## Motivation
Config-1 kernels are effectively singular (e.g. `cond(A_l) ≈ 9.4×10²³`, `rank=501/512`), which explains the catastrophic errors observed in tasks b–d. Task e explores a more robust reconstruction by:

1. **Diagnosing kernels** with condition numbers and ranks.
2. **Solving the least-squares formulation** `min_X ‖A_l X A_r - B‖_F` via two sequential normal equations (without adding explicit regularization).

## Method
We solve the Frobenius-norm least-squares problem via two sequential least-squares subproblems:

```
min_Y ||A_l Y - B||_F²
min_X ||A_r^T X^T - Y^T||_F²
```

Each block is solved with `np.linalg.lstsq` (effectively forming the Moore–Penrose pseudo-inverse) to suppress the impact of rank deficiency without manual λ tuning.

## Results (relative Frobenius error / PSNR)

| Image | Config | LU | QR | LS (no λ) | Notes |
|-------|--------|----|----|-----------|-------|
| 512_car | config1 | 3.2×10⁶⁶ / -1276 dB | 3.1×10⁵⁷ / -1095 dB | **7.9×10⁻² / 76.3 dB** | Plain LS already stabilizes the singular kernels |
| 1024_books | config1 | 1.5×10¹⁴⁰ / -2748 dB | 4.3×10¹²⁹ / -2538 dB | **7.1×10⁻² / 78.1 dB** | Same improvement; no explicit regularizer needed |
| 512_car | config2 | 1.3×10⁻⁸ / ∞ | 1.3×10⁻⁸ / ∞ | 2.8×10⁻⁸ / ∞ | LS matches baseline accuracy |
| 1024_books | config2 | 1.2×10⁻⁷ / 193.5 dB | 1.3×10⁻⁷ / 192.9 dB | 2.1×10⁻⁷ / 188.7 dB | Slightly worse than QR/LU but still close |

Runtime: The dense least-squares solve takes ≈1.4 s (512²) and ≈7.4 s (1024²), slower than LU/QR but acceptable given the dramatic accuracy gain on config1.

## Observations
1. **Kernel quality matters**: Config1 matrices lose ~2–3% of rank, making direct inversion meaningless; pseudo-inverse based LS mitigates this automatically.
2. **Plain LS is enough**: Even without explicit Tikhonov terms, the least-squares solution suppresses the blow-ups and yields PSNR ≈ 75 dB on singular kernels.
3. **Well-conditioned cases remain accurate**: For config2 the LS solution matches LU/QR up to rounding noise, though it can be marginally worse on the 1024 case.
4. **Computational cost**: Dense lstsq is slower than LU/QR but still acceptable; exploiting kernel structure could bring it closer to real-time.

## Conclusion
Task e demonstrates that augmenting the deblurring workflow with condition-number diagnostics plus a pseudo-inverse least-squares solver dramatically improves reconstructions for ill-conditioned kernels, while highlighting the trade-off between stability and runtime for well-conditioned cases. Future work could incorporate banded/structured LS solvers or iterative Krylov methods to accelerate the pseudoinverse step.

