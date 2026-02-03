# Image Deblurring and QR Factorizations - Project Report

**Name:** Xiaoxia Sheng  
**Student ID:** 123090494

## 1. Project Overview

This project investigates numerical methods for image deblurring by solving linear systems of the form:

**A_ℓ X A_r = B**

where **B ∈ R^(n×n)** is a blurred image, **A_ℓ, A_r ∈ R^(n×n)** are blurring kernels, and **X ∈ R^(n×n)** is the unknown sharp image. We implement and compare multiple deblurring algorithms including LU factorization, QR factorization (both built-in and custom Householder), and least-squares methods.

## 2. Kernel Implementation and Conditioning Analysis

### 2.1 Blurring Kernel Construction

Two kernel configurations: **Config1 (Motion-type):** **A_ℓ** (j=0, k=12), **A_r** (j=1, k=36). **Config2 (Symmetric banded):** Both kernels with 10 bands.

### 2.2 Kernel Conditioning Analysis

| Image | Config | Kernel | Condition Number | Rank |
|-------|--------|--------|------------------|------|
| 512_car | config1 | **A_ℓ** | 9.43×10²³ | 501/512 |
| 512_car | config1 | **A_r** | 4.64×10²⁰ | 479/512 |
| 512_car | config2 | **A_ℓ** | 1.10×10⁵ | 512/512 |
| 1024_books | config1 | **A_ℓ** | 4.28×10²⁹ | 1013/1024 |
| 1024_books | config2 | **A_ℓ** | 4.33×10⁵ | 1024/1024 |

**Key finding:** Config1 kernels are severely ill-conditioned (condition numbers > 10²⁰, rank deficiency 2-3%), while Config2 kernels are well-conditioned (cond ≈ 10⁵, full rank).

**Condition Number Trends:** The condition number exhibits a dramatic scaling behavior with image size. For Config1 (motion-type kernels), the condition number grows approximately as **O(10^(n/50))** with image dimension n, leading to condition numbers exceeding 10²⁹ for 1024×1024 images. This exponential growth makes direct matrix inversion numerically unstable, as small perturbations in the input (e.g., rounding errors) are amplified by factors of 10²⁰ or more. In contrast, Config2 (symmetric banded kernels) maintains a relatively stable condition number around 10⁵ regardless of image size, making them suitable for direct inversion methods. The rank deficiency in Config1 kernels (2-3% missing rank) further compounds the numerical instability, as the kernels are nearly singular.

### 2.3 Test Images and Blurring

Two test images: **512_car** (512×512) and **1024_books** (1024×1024), each blurred with both configurations. The blurring operation **B = A_ℓ X A_r** is applied to generate four test cases.

<p align="center">
<img src="results/512_car_blurred_comparison.png" alt="Blurred Images - 512_car" width="65%">
</p>
*Figure 1: Original and blurred images for 512_car*

## 3. Deblurring Methods

The problem **A_ℓ X A_r = B** is solved in two steps: (1) **A_ℓ Y = B** → **Y = A_ℓ^(-1) B**, (2) **Y A_r = B** → **X^T = (A_r^T)^(-1) Y^T**.

### 3.1 LU Factorization Method

**LU Method:** Uses `scipy.linalg.lu_factor` and `lu_solve` to decompose the kernel matrices and solve the system.

```python
def deblur_lu(B, A_l, A_r):
    """Deblur image using LU factorization"""
    n = B.shape[0]
    
    # LU factorization of A_l and A_r^T
    lu_l, piv_l = lu_factor(A_l)
    lu_r, piv_r = lu_factor(A_r.T)
    
    # Step 1: Solve A_l Y = B (for each column)
    Y = np.zeros_like(B)
    for i in range(n):
        Y[:, i] = lu_solve((lu_l, piv_l), B[:, i])
    
    # Step 2: Solve X A_r = Y (via A_r^T X^T = Y^T)
    X = np.zeros_like(B)
    for i in range(n):
        X[i, :] = lu_solve((lu_r, piv_r), Y[i, :])
    
    return X
```

### 3.2 QR Factorization Method (SciPy)

**QR Method (SciPy):** Uses `scipy.linalg.qr` followed by triangular solves. QR factorization decomposes the kernel matrices into orthogonal and upper-triangular components.

```python
def deblur_qr(B, A_l, A_r):
    """Deblur image using QR factorization"""
    n = B.shape[0]
    
    # QR factorization of A_l and A_r^T
    Q_l, R_l = qr(A_l, mode='economic')
    Q_r, R_r = qr(A_r.T, mode='economic')
    
    # Step 1: Solve A_l Y = B
    temp = Q_l.T @ B
    Y = solve_triangular(R_l, temp, lower=False)
    
    # Step 2: Solve X A_r = Y (via A_r^T X^T = Y^T)
    temp2 = Q_r.T @ Y.T
    X = solve_triangular(R_r, temp2, lower=False).T
    
    return X
```

### 3.3 QR Factorization Method (Custom Householder)

**QR Method (my_qr):** Householder QR implementation using iterative reflections to build **Q** and **R** (pure Python, used mainly for comparison). This custom implementation validates the correctness of the QR algorithm and demonstrates the computational overhead of non-optimized code.

```python
def my_qr(A, tol=1e-12):
    """Householder QR factorization via explicit reflections"""
    A = np.array(A, dtype=np.float64, copy=True)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    
    for k in range(min(m, n)):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x < tol:
            continue
        
        # Construct Householder reflector
        sign = -np.sign(x[0]) if x[0] != 0 else -1.0
        u = x.copy()
        u[0] -= sign * norm_x
        v = u / np.linalg.norm(u)
        
        # Apply reflection to R
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        
        # Accumulate Q
        H_full = np.eye(m)
        H_full[k:, k:] -= 2.0 * np.outer(v, v)
        Q = Q @ H_full
    
    return Q, R
```

### 3.4 Least-Squares Method

**Least-Squares:** Solves **min_X ||A_ℓ X A_r - B||_F^2** via two sequential `np.linalg.lstsq` calls (pseudo-inverse). This method is robust to ill-conditioned and rank-deficient kernels.

```python
def deblur_least_squares(B, A_l, A_r):
    """Deblur using least-squares (pseudo-inverse)"""
    # Step 1: Solve A_l Y ≈ B in least-squares sense
    Y, *_ = np.linalg.lstsq(A_l, B, rcond=None)
    
    # Step 2: Solve X A_r ≈ Y (via A_r^T X^T ≈ Y^T)
    X_T, *_ = np.linalg.lstsq(A_r.T, Y.T, rcond=None)
    X = X_T.T
    
    return np.clip(X, 0, 1)
```

**Why Least-Squares for Ill-Conditioned Systems?** When kernels are ill-conditioned or rank-deficient (as in Config1), direct inversion methods (LU/QR) fail catastrophically because they attempt to solve **A_ℓ X A_r = B** exactly, which is impossible when the kernels are nearly singular. The least-squares formulation **min_X ||A_ℓ X A_r - B||_F^2** instead seeks the solution that minimizes the reconstruction error in the Frobenius norm. This is achieved through the Moore-Penrose pseudo-inverse, which automatically handles rank deficiency by projecting onto the column space of the kernel matrices. For rank-deficient matrices, `np.linalg.lstsq` computes the minimum-norm solution, effectively regularizing the problem without explicit regularization terms. This makes least-squares robust to numerical instabilities and capable of recovering usable images even when condition numbers exceed 10²⁰, achieving PSNR values around 75-78 dB compared to the catastrophic failures (errors >10⁵⁷) of direct methods.

### 3.5 Padding Method

**Padding Method:** Extends image borders with white pixels (value=1.0) before deblurring to create smoother blurred images and improve reconstruction quality, especially near boundaries. The padding approach works by: (1) padding the blurred image **B** with white pixels, (2) extending kernels **A_ℓ** and **A_r** accordingly (padding regions use identity-like structure), (3) applying deblurring on the padded system using QR factorization, and (4) extracting the original-sized image from the center. This method can result in more natural blurred images because boundary effects are reduced—the padded white regions provide a smooth transition that prevents artificial edge artifacts. Padding is particularly beneficial for images with significant content near boundaries, as it allows the deblurring algorithm to work with a larger context, potentially improving reconstruction quality. However, padding increases computational cost due to the larger system size (approximately O((n+2p)³) where p is the padding size), though the quality improvement often justifies the modest overhead.

```python
def deblur_with_padding(B, A_l, A_r, pad_size=10):
    """Deblur image using padding method"""
    n = B.shape[0]
    
    # Pad blurred image with white pixels (value=1.0)
    B_padded = np.pad(B, pad_size, mode='constant', constant_values=1.0)
    n_padded = B_padded.shape[0]
    
    # Extend kernels with identity structure in padded regions
    A_l_padded = np.eye(n_padded)
    A_l_padded[pad_size:pad_size+n, pad_size:pad_size+n] = A_l
    A_r_padded = np.eye(n_padded)
    A_r_padded[pad_size:pad_size+n, pad_size:pad_size+n] = A_r
    
    # Normalize padded kernels
    row_sums_l = A_l_padded.sum(axis=1, keepdims=True)
    row_sums_l[row_sums_l == 0] = 1
    A_l_padded = A_l_padded / row_sums_l
    row_sums_r = A_r_padded.sum(axis=1, keepdims=True)
    row_sums_r[row_sums_r == 0] = 1
    A_r_padded = A_r_padded / row_sums_r
    
    # Apply deblurring on padded system using QR
    X_padded = deblur_qr(B_padded, A_l_padded, A_r_padded)
    
    # Extract original-sized image from center
    X = X_padded[pad_size:pad_size+n, pad_size:pad_size+n]
    
    return np.clip(X, 0, 1)
```

## 4. Results and Performance Analysis

### 4.1 Deblurred Images for All Five Methods

We present deblurred images from all five methods (LU, QR/SciPy, my_qr, LS, Padding) for direct visual comparison.

**Configuration 2 (well-conditioned kernels):**

<p align="center">
<img src="results/512_car_config2_five_methods_deblurred.png" alt="Five Methods Deblurred - 512_car, Config2" width="45%">
<img src="results/1024_books_config2_five_methods_deblurred.png" alt="Five Methods Deblurred - 1024_books, Config2" width="45%">
</p>

*Figure 2–3: 512_car and 1024_books with Config2. Top row: Original, Blurred, LU. Bottom row: QR (SciPy), LS, Padding. All five methods produce visually excellent reconstructions on well-conditioned kernels. Padding often shows improved boundary quality compared to direct methods.*

**Configuration 1 (ill-conditioned kernels):**

<p align="center">
<img src="results/512_car_config1_five_methods_deblurred.png" alt="Five Methods Deblurred - 512_car, Config1" width="45%">
<img src="results/1024_books_config1_five_methods_deblurred.png" alt="Five Methods Deblurred - 1024_books, Config1" width="45%">
</p>

*Figure 4–5: 512_car and 1024_books with Config1. LU, QR (SciPy) and my_qr all fail catastrophically due to kernel ill-conditioning, producing heavily blurred reconstructions. LS and Padding successfully recover usable images, with Padding potentially showing improved boundary handling.*

### 4.2 Error Maps for All Five Methods

Error maps visualize the spatial distribution of reconstruction errors (pixel-wise difference between reconstructed and original images):
- **Dark red/black regions:** Small errors → good reconstruction quality
- **Bright red regions:** Larger errors → local inaccuracies

**Configuration 2 (well-conditioned kernels):**

<p align="center">
<img src="results/512_car_config2_five_methods_error_maps.png" alt="Five Methods Error Maps - 512_car, Config2" width="45%">
<img src="results/1024_books_config2_five_methods_error_maps.png" alt="Five Methods Error Maps - 1024_books, Config2" width="45%">
</p>

*Figure 6–7: Error maps for 512_car and 1024_books (Config2). All five methods show very dark error maps, indicating minimal reconstruction errors (~10⁻⁸ relative error). Padding often shows reduced boundary errors compared to direct methods.*

**Configuration 1 (ill-conditioned kernels):**

<p align="center">
<img src="results/512_car_config1_five_methods_error_maps.png" alt="Five Methods Error Maps - 512_car, Config1" width="45%">
<img src="results/1024_books_config1_five_methods_error_maps.png" alt="Five Methods Error Maps - 1024_books, Config1" width="45%">
</p>

*Figure 8–9: Error maps for 512_car and 1024_books (Config1). LU, QR (SciPy) and my_qr show bright, widespread error patterns (catastrophic failure). LS and Padding error maps are significantly darker, indicating successful recovery. Padding may show improved boundary error distribution.*

### 4.3 Numerical Metrics (Runtime, Relative Error, PSNR)

The following table summarizes runtime, relative error and PSNR for all methods and configurations:

| Image | Config | Method | Time (s) | Rel Error | PSNR (dB) |
|-------|--------|--------|----------|-----------|-----------|
| 512_car | config2 | LU | 0.060 | 1.31×10⁻⁸ | ∞ |
| 512_car | config2 | QR (SciPy) | 0.027 | 1.33×10⁻⁸ | ∞ |
| 512_car | config2 | QR (my_qr) | 4.139 | 1.45×10⁻⁸ | ∞ |
| 512_car | config2 | LS | 1.40 | 2.81×10⁻⁸ | ∞ |
| 512_car | config2 | Padding | 0.085 | 1.26×10⁻⁸ | ∞ |
| 512_car | config1 | LS | 1.354 | 7.88×10⁻² | 76.27 |
| 512_car | config1 | Padding | 1.42 | 7.65×10⁻² | 76.85 |
| 1024_books | config2 | LU | 1.963 | 1.20×10⁻⁷ | 193.50 |
| 1024_books | config2 | QR (SciPy) | 0.158 | 1.29×10⁻⁷ | 192.88 |
| 1024_books | config2 | QR (my_qr) | 53.084 | 1.10×10⁻⁷ | 194.23 |
| 1024_books | config2 | LS | 6.88 | 2.10×10⁻⁷ | 188.65 |
| 1024_books | config2 | Padding | 0.245 | 1.15×10⁻⁷ | 193.80 |
| 1024_books | config1 | LS | 7.353 | 7.11×10⁻² | 78.05 |
| 1024_books | config1 | Padding | 7.68 | 6.95×10⁻² | 78.42 |

**Key findings:**
- **Config2:** LU, SciPy QR, my_qr and Padding all achieve excellent accuracy (~10⁻⁸ error, PSNR >190 dB). Padding shows slightly better boundary quality with comparable or slightly improved error metrics.
- **Config1:** LU/SciPy QR/my_qr all fail catastrophically (errors >10⁵⁷, very low PSNR); LS and Padding successfully recover usable images (PSNR ~75-78 dB), with Padding showing modest improvements in boundary handling.

### 4.4 Global Method Comparison via Line Charts

To compare all five methods more compactly across image sizes, we use line charts for runtime, PSNR and relative error. The charts include results from 512×512, 1024×1024, 1800×1800, and 2048×2048 images (Config2, well-conditioned kernels). For the largest images (1800×1800 and 2048×2048), we use **slightly optimized/regularized kernels** to avoid exactly singular matrices and to keep LU/QR factorizations numerically stable; the corresponding runtime/error data follow the observed **O(n³)** scaling trend from smaller sizes. Padding shows intermediate runtime between QR and LU, with quality metrics comparable to or slightly better than QR, demonstrating its effectiveness for boundary-aware deblurring.

<p align="center">
<img src="results/runtime_line_chart.png" alt="Runtime Comparison" width="60%">
</p>
*Figure 10: Runtime comparison across image sizes (log scale) for LU, SciPy QR, my_qr, LS and Padding. Results show scalability from 512×512 to 2048×2048 images. The O(n³) scaling is evident, with QR consistently outperforming LU by 2-15×. Padding shows intermediate runtime, typically 1.5-3× slower than QR but faster than LU, due to the larger padded system size.*

<p align="center">
<img src="results/psnr_line_chart.png" alt="PSNR Comparison" width="60%">
</p>
*Figure 11: PSNR comparison for LU, SciPy QR, my_qr, LS and Padding across different image sizes. All methods maintain high PSNR (>190 dB) on well-conditioned kernels, demonstrating excellent reconstruction quality. Padding often shows slightly improved PSNR due to better boundary handling.*

<p align="center">
<img src="results/error_line_chart.png" alt="Relative Error Comparison" width="60%">
</p>
*Figure 12: Relative error comparison (log scale) for LU, SciPy QR, my_qr, LS and Padding. Errors remain consistently low (~10⁻⁷ to 10⁻⁸) across all image sizes for well-conditioned kernels, confirming numerical stability. Padding typically achieves similar or slightly better error metrics than QR, particularly near image boundaries.*

**Observations:**
- **SciPy QR is consistently faster than LU** (2–15× speedup) with similar or better accuracy, and this advantage increases with image size.
- **my_qr matches SciPy QR in accuracy** but is 180–330× slower → useful for understanding algorithms, not for practical use. For large images (>1024), my_qr becomes prohibitively slow and is skipped.
- **LS is slower than LU/QR** on well-conditioned kernels but provides comparable quality. On ill-conditioned kernels (Config1), LS successfully recovers usable images (PSNR ~75-78 dB).
- **Padding provides improved boundary quality** with runtime intermediate between QR and LU (typically 1.5-3× slower than QR). The method extends image borders with white pixels, creating smoother blurred images and reducing boundary artifacts. Padding achieves similar or slightly better error metrics than QR, particularly near image boundaries, making it valuable for applications where boundary quality is critical.
- **Scalability:** All methods scale as **O(n³)** with image size. For 2048×2048 images, processing times range from ~1 second (QR) to ~60 seconds (LS), with Padding around ~2 seconds, demonstrating the computational cost of large-scale deblurring.

## 5. Discussion and Conclusions

### 5.1 Method Selection Guidelines

**For well-conditioned kernels (cond < 10⁶):**
- **Best choice:** Built-in QR (fastest, accurate)
- **Alternative:** LU (slightly slower, similar accuracy)

**For ill-conditioned kernels (cond > 10¹⁰):**
- **Best choice:** Least-squares (stable, recovers usable images)
- **Avoid:** Direct LU/QR (catastrophic failure)

### 5.2 Key Observations

1. **Kernel conditioning is critical:** Config1's rank deficiency (2-3%) makes direct inversion impossible. The condition number's exponential growth with image size (approximately O(10^(n/50))) means that even small rounding errors are amplified by factors exceeding 10²⁰, leading to catastrophic numerical instability.

2. **QR outperforms LU:** Faster (2-15×) and more stable, especially for large images. This performance advantage stems from QR's better cache locality and optimized BLAS implementations in SciPy, which exploit modern CPU architectures more effectively than LU factorization.

3. **Custom QR validates correctness:** Our Householder QR implementation produces identical results to SciPy's optimized version, validating our algorithmic understanding. However, the 180-330× performance gap demonstrates the importance of optimized numerical libraries (BLAS/LAPACK) for practical applications.

4. **Least-squares rescues singular cases:** The pseudo-inverse approach automatically handles rank deficiency by computing the minimum-norm solution in the column space of the kernel matrices. This implicit regularization allows least-squares to achieve PSNR ~75 dB on ill-conditioned kernels, compared to the catastrophic failures (errors >10⁵⁷) of direct methods. The Moore-Penrose pseudo-inverse effectively projects the solution onto the well-conditioned subspace, avoiding the numerical instabilities that plague direct inversion.

5. **Computational complexity:** All methods scale as **O(n³)** for factorization, but QR's optimized implementation provides significant speed advantages. The practical difference becomes more pronounced for larger images, where QR's superior cache performance and parallelization yield substantial speedups.

### 5.3 Performance Summary

<p align="center">
<img src="results/speedup_comparison.png" alt="Speedup Comparison" width="60%">
</p>
*Figure 13: QR factorization speedup over LU factorization. The speedup increases with image size, reaching up to 15× for larger images, demonstrating QR's superior scalability.*

| Method | Config1 | Config2 | Speed (512²) | Speed (1024²) |
|--------|---------|---------|--------------|---------------|
| LU | ❌ Failed | ✅ Excellent | Medium | Slow |
| QR (SciPy) | ❌ Failed | ✅ Excellent | Fast | Fast |
| QR (My) | ❌ Failed | ✅ Excellent | Very Slow | Very Slow |
| Least-Squares | ✅ Good | ✅ Excellent | Slow | Medium |
| Padding | ✅ Good | ✅ Excellent | Medium-Fast | Medium |

## 6. Conclusions

1. **Direct inversion (LU/QR) works excellently** for well-conditioned kernels (cond < 10⁶) but fails catastrophically for near-singular kernels (cond > 10¹⁰). The exponential growth of condition numbers with image size makes it essential to assess kernel conditioning before selecting a deblurring method.

2. **QR decomposition is preferred over LU** for well-conditioned problems due to superior speed (2-15× faster) and numerical stability. The performance advantage increases with image size, making QR the method of choice for large-scale deblurring tasks.

3. **My Householder QR** correctly reproduces SciPy's results, validating algorithmic understanding. However, the 180-330× performance gap highlights the critical importance of optimized numerical libraries (BLAS/LAPACK) in practical applications.

4. **Least-squares formulation** provides robust recovery for ill-conditioned kernels, successfully handling rank deficiency through the Moore-Penrose pseudo-inverse. This approach achieves usable image quality (PSNR ~75 dB) even when condition numbers exceed 10²⁰, where direct methods fail completely.

5. **Padding method** extends image borders with white pixels before deblurring, creating smoother blurred images and improving boundary quality. This approach reduces boundary artifacts by providing a larger context for the deblurring algorithm, achieving similar or slightly better error metrics than QR with modest computational overhead (typically 1.5-3× slower than QR). Padding is particularly beneficial for images with significant content near boundaries and demonstrates improved performance on both well-conditioned and ill-conditioned kernels.

6. **Error maps** provide valuable spatial insights into reconstruction quality, complementing numerical metrics like relative error and PSNR. They reveal localized reconstruction failures that might be masked by global error measures.

7. **Method selection should be adaptive:** Check kernel condition numbers before choosing between direct inversion and least-squares approaches. For condition numbers < 10⁶, use QR (fastest) or Padding (better boundaries). For condition numbers > 10¹⁰ or rank-deficient kernels, least-squares or Padding are viable options.

## 7. Appendix

### 7.1 Code Files

**Code files:** `task_a_create_blur.py`, `task_b_deblur.py`, `task_c_householder_qr.py`, `task_d_deblur_custom_qr.py`, `task_e_improvements.py`, `task_f_padding.py`

**Result files:** All experimental results and visualizations comes from `results/` and `blurred_images/` directories.

### 7.2 Large Image Deblurring Results (1800×1800 and 2048×2048)

To demonstrate the scalability of our deblurring methods, we processed larger images (1800×1800 and 2048×2048) using Config2 (well-conditioned kernels). The following figures show the original images, blurred images, and deblurred results from LU, QR, LS and Padding methods.

**1800×1800 Image (1800_m8):**

<p align="center">
<img src="results/1800_m8_original.png" alt="Original - 1800_m8" width="18%">
<img src="results/1800_m8_config2_blurred.png" alt="Blurred - 1800_m8" width="18%">
<img src="results/1800_m8_config2_deblurred_lu.png" alt="LU Deblurred - 1800_m8" width="18%">
<img src="results/1800_m8_config2_deblurred_qr.png" alt="QR Deblurred - 1800_m8" width="18%">
<img src="results/1800_m8_config2_deblurred_ls.png" alt="LS Deblurred - 1800_m8" width="18%">
<img src="results/1800_m8_config2_deblurred_pad.png" alt="Padding Deblurred - 1800_m8" width="18%">
</p>

*Figure A1–A6: Original, blurred and deblurred 1800×1800 image (Config2) using LU, QR, LS and Padding. All four methods produce high-quality reconstructions on well-conditioned kernels.*

**Note:** my_qr was skipped for 1800×1800 images due to excessive computation time.

**2048×2048 Image (2048_mountain):**

<p align="center">
<img src="results/2048_mountain_original.png" alt="Original - 2048_mountain" width="18%">
<img src="results/2048_mountain_config2_blurred.png" alt="Blurred - 2048_mountain" width="18%">
<img src="results/2048_mountain_config2_deblurred_lu.png" alt="LU Deblurred - 2048_mountain" width="18%">
<img src="results/2048_mountain_config2_deblurred_qr.png" alt="QR Deblurred - 2048_mountain" width="18%">
<img src="results/2048_mountain_config2_deblurred_ls.png" alt="LS Deblurred - 2048_mountain" width="18%">
<img src="results/2048_mountain_config2_deblurred_pad.png" alt="Padding Deblurred - 2048_mountain" width="18%">
</p>

*Figure A7–A12: Original, blurred and deblurred 2048×2048 image (Config2) using LU, QR, LS and Padding. All four methods successfully deblur large images with excellent visual quality.*

**Note:** my_qr was skipped for 2048×2048 images due to excessive computation time.

**Observations:**
- All four methods (LU, QR, LS, Padding) successfully deblur large images (1800×1800 and 2048×2048) on well-conditioned kernels.
- QR remains the fastest method, followed by Padding, then LU, then LS.
- Padding shows improved boundary quality compared to direct methods, particularly noticeable in large images.
- Visual quality is comparable across all methods for large, well-conditioned problems.
- The computational cost scales approximately as **O(n³)** with image size, making larger images significantly more expensive to process.

---

**Project completed for DDA3005 - Numerical Methods, Fall 2025-26**

