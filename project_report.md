# DDA3005 Course Project — Image Deblurring and QR Factorizations

## 1. Project Overview
The goal of this project is to study numerical methods for image deblurring, focusing on linear systems of the form
\[
A_\ell X A_r = B,
\]
where \(B\) is a blurred image, \(A_\ell, A_r\) are left/right blur kernels, and \(X\) is the unknown sharp image. The work spans data generation, direct solvers based on LU/QR, custom Householder QR, and least-squares formulations that remain stable when the blur kernels are nearly singular.


## 2. Implemented Components

### 2.1 Data generation (Task a)
- Implemented `task_a_create_blur.py`, which:
  - Loads grayscale versions of the provided test images (e.g., `test_images/512_car_original.png`, `test_images/1024_books_original.png`).
  - Builds two families of blur kernels: the motion-type kernels described in the project handout (config1) and symmetric banded kernels (config2).
  - Applies \(B = A_\ell X A_r\) to produce blurred images and saves all intermediates to `blurred_images/`.
  - Produces visual comparisons (original vs. blurred) under `results/`.

### 2.2 Direct deblurring with LU/QR (Task b)
- `task_b_deblur.py` solves \(A_\ell X A_r = B\) by:
  - LU factorization on both kernels (`scipy.linalg.lu_factor/lu_solve`).
  - Built-in QR factorization (`scipy.linalg.qr`) followed by triangular solves.
- Metrics reported: runtime, relative Frobenius error, PSNR, plus visualizations of reconstructions and error maps.
- Results are summarized in `task_b_summary.md`.

### 2.3 Custom Householder QR (Task c)
- `task_c_householder_qr.py` implements a full Householder QR without column pivoting:
  - Generates reflectors explicitly, accumulates orthogonal matrix \(Q\), returns \(Q, R\).
  - Includes a self-test routine comparing against NumPy’s QR.

### 2.4 Re-running Task b with custom QR (Task d)
- `task_d_deblur_custom_qr.py` plugs `my_qr` into the deblurring pipeline while still retaining LU and SciPy-QR baselines.
- Produces expanded comparisons (LU vs. SciPy-QR vs. custom-QR) and detailed observations stored in `task_d_summary.md`.

### 2.5 Least-squares improvements (Task e)
- `task_e_improvements.py` diagnoses blur-kernel conditioning (condition numbers + ranks) and solves the Frobenius least-squares problem using two sequential pseudo-inverse solves (`np.linalg.lstsq`) without explicit regularization.
- Demonstrates that pseudo-inverse LS dramatically improves reconstructions for the nearly singular config1 kernels.
- Findings documented in `task_e_summary.md`.


## 3. Experimental Setup
- **Images:** Two representative resolutions (512×512 car, 1024×1024 books) with two blur settings:
  - **config1:** Motion-type kernels from the project description, which turn out to be rank-deficient (e.g., `rank(A_\ell)=501/512`).
  - **config2:** Symmetric 10-band kernels, well conditioned (`cond ≈ 1e5`).
- All computations use double precision. Runtime measurements were taken on the provided Windows environment (Anaconda Python, SciPy 1.10 API).
- Metrics: relative Frobenius error \( \|X_{\text{rec}}-X\|_F / \|X\|_F \) and PSNR (with images normalized to [0,1]).
- Visual artifacts and `.npy` arrays can be found under `results/`.


## 4. Results and Observations

### 4.1 Task b (LU vs. SciPy-QR)
- Config2 (well conditioned): both LU and QR reach errors near \(10^{-8}\)–\(10^{-7}\) with PSNR > 190 dB or infinity; QR is ~2× faster for 512 images and ~13× faster for 1024 images.
- Config1 (ill conditioned): both methods fail (errors \(10^{57}\)–\(10^{140}\), PSNR below –1000 dB) because the kernels are almost singular.

### 4.2 Task d (Custom QR)
- Custom Householder QR matches the accuracy of SciPy’s QR on config2 but is 180–330× slower due to explicit dense operations in Python.
- On config1 it still fails because the underlying kernels are singular, though it avoids numerical crashes thanks to guarded least-squares solves.

### 4.3 Task e (Least Squares without explicit λ)
- Simply solving the least-squares problems via `np.linalg.lstsq` mitigates the kernel singularity:

| Image | Config | LU (RelErr/PSNR) | QR (RelErr/PSNR) | LS (RelErr/PSNR) |
|-------|--------|------------------|------------------|------------------|
| 512_car | config1 | \(3.2×10^{66}\) / –1276 dB | \(3.1×10^{57}\) / –1095 dB | **\(7.9×10^{-2}\) / 76.3 dB** |
| 1024_books | config1 | \(1.5×10^{140}\) / –2748 dB | \(4.3×10^{129}\) / –2538 dB | **\(7.1×10^{-2}\) / 78.1 dB** |
| 512_car | config2 | \(1.3×10^{-8}\) / ∞ | \(1.3×10^{-8}\) / ∞ | \(2.8×10^{-8}\) / ∞ |
| 1024_books | config2 | \(1.2×10^{-7}\) / 193.5 dB | \(1.3×10^{-7}\) / 192.9 dB | \(2.1×10^{-7}\) / 188.7 dB |

- On config1, LS yields visually good reconstructions with PSNR ≈ 75 dB.
- On config2, LS performs similarly but not better than LU/QR, confirming that regularization is unnecessary when the kernels are well conditioned.
- Runtime: LS takes ≈1.4 s (512²) and ≈7.4 s (1024²), slower than LU/QR but acceptable given the stability gains.


## 5. Conclusions
1. **Kernel conditioning drives accuracy:** Config1 kernels lose 2–3% of rank and trigger enormous errors in direct LU/QR solves. Config2 is well conditioned and easy to invert.
2. **Built-in QR is the fastest reliable direct solver:** For well-behaved kernels it outperforms LU in runtime while retaining accuracy.
3. **Custom Householder QR validates our understanding:** It reproduces SciPy’s results but highlights the cost of naive dense implementations.
4. **Least-squares (pseudo-inverse) solves rescue singular cases:** Without explicit regularization, `np.linalg.lstsq` already stabilizes the reconstruction and delivers usable images for config1.
5. **Trade-offs:** LU/QR are ideal for well-conditioned kernels; LS is essential when kernels are nearly singular, albeit at higher computational cost.


## 6. Future Work
- Develop adaptive logic to choose between LU/QR and LS based on measured condition numbers.
- Exploit the banded/Toeplitz structure of the kernels to accelerate both QR and least-squares solvers.
- Extend the pipeline to color images and larger kernels, possibly leveraging FFT-based deconvolution or iterative Krylov methods (CG/LSQR) for very large systems.
- Integrate peak-signal-to-noise ratio (PSNR) monitoring into the runtime to automatically flag unstable reconstructions.


## 7. Appendix / Artefacts
- Code:
  - `task_a_create_blur.py`, `task_b_deblur.py`, `task_c_householder_qr.py`, `task_d_deblur_custom_qr.py`, `task_e_improvements.py`.
- Summaries:
  - `task_b_summary.md`, `task_d_summary.md`, `task_e_summary.md`.
- Visuals and reconstructions: `results/` (PNG + NPY files for each configuration).
- Blurred data and kernels: `blurred_images/`.

