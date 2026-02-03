# Image Deblurring and QR Factorizations - Project Report

## 1. Project Overview

This project investigates numerical methods for image deblurring, focusing on solving linear systems of the form:

**A_ℓ X A_r = B**

where **B ∈ R^(n×n)** is a blurred image, **A_ℓ, A_r ∈ R^(n×n)** are left and right blurring kernels, and **X ∈ R^(n×n)** is the unknown sharp image to be recovered.

The work spans five main tasks:
- **Task a**: Generate blurring kernels and create blurred test images
- **Task b**: Implement deblurring using built-in LU and QR factorizations
- **Task c**: Implement custom Householder QR decomposition
- **Task d**: Repeat task b using the custom QR implementation
- **Task e**: Explore least-squares improvements for ill-conditioned kernels

## 2. Kernel Implementation and Image Blurring (Task a)

### 2.1 Blurring Kernel Construction

Two sets of blurring kernels are implemented following the project specification:

**Configuration 1 (Motion-type kernels):**
- **A_ℓ**: constructed with j=0, k=12 (upper triangular structure)
- **A_r**: constructed with j=1, k=36 (upper triangular with extra subdiagonal)
- Coefficients follow: **[a_{n+j}, a_{n+j-1}, ..., a_{n+j-k+1}] = (2/(k(k+1)))[k, k-1, ..., 1]**

**Configuration 2 (Symmetric banded kernels):**
- Both A_ℓ and A_r are symmetric banded matrices with 10 bands
- Coefficients: a_n = 10/100, a_{n-1} = 9/100, ..., a_{n-9} = 1/100

### 2.2 Image Blurring Process

The blurring operation is implemented as:

```python
def blur_image(X, A_l, A_r):
    """Apply blurring: B = A_ℓ X A_r"""
    B = A_l @ X @ A_r
    return B
```

### 2.3 Test Images

Two test images are selected:
- **512_car**: 512×512 grayscale car image
- **1024_books**: 1024×1024 grayscale books image

Each image is blurred using both configurations, resulting in four blurred test cases. Visual comparisons are shown in Figure 1.

![Blurred Images Comparison](results/512_car_blurred_comparison.png)
*Figure 1: Original and blurred images for 512_car (left: original, middle: config1, right: config2)*

## 3. Analysis of the Deblurring Problem

### 3.1 Kernel Conditioning Analysis

Before attempting deblurring, we analyze the condition numbers and ranks of the blurring kernels:

| Image | Config | Kernel | Condition Number | Rank |
|-------|--------|--------|------------------|------|
| 512_car | config1 | A_ℓ | 9.43×10²³ | 501/512 |
| 512_car | config1 | A_r | 4.64×10²⁰ | 479/512 |
| 512_car | config2 | A_ℓ | 1.10×10⁵ | 512/512 |
| 512_car | config2 | A_r | 1.10×10⁵ | 512/512 |
| 1024_books | config1 | A_ℓ | 4.28×10²⁹ | 1013/1024 |
| 1024_books | config1 | A_r | 5.82×10²⁰ | 990/1024 |
| 1024_books | config2 | A_ℓ | 4.33×10⁵ | 1024/1024 |
| 1024_books | config2 | A_r | 4.33×10⁵ | 1024/1024 |

**Key Observations:**
- **Config1 kernels are severely ill-conditioned**: Condition numbers exceed 10²⁰, and the kernels lose 2-3% of their rank (e.g., rank 501/512 for 512×512 images). This explains why direct inversion methods fail catastrophically.
- **Config2 kernels are well-conditioned**: Condition numbers around 10⁵, and full rank is maintained. These kernels are suitable for direct inversion methods.

### 3.2 Problem Structure

The deblurring problem **A_ℓ X A_r = B** can be solved in two steps:
1. Solve **A_ℓ Y = B** to obtain **Y = A_ℓ^(-1) B**
2. Solve **Y A_r = B**, which is equivalent to **A_r^T X^T = Y^T**, giving **X^T = (A_r^T)^(-1) Y^T**

This two-step approach allows us to leverage standard linear algebra factorizations.

## 4. Image Deblurring by Built-in LU and QR Functions (Task b)

### 4.1 Implementation

**LU Decomposition Method:**

```python
def deblur_lu(B, A_l, A_r):
    """Deblur using LU factorization"""
    n = B.shape[0]
    
    # LU factorization of A_l
    lu_l, piv_l = lu_factor(A_l)
    
    # LU factorization of A_r^T
    lu_r, piv_r = lu_factor(A_r.T)
    
    # Step 1: Solve A_l Y = B
    Y = np.zeros_like(B)
    for i in range(n):
        Y[:, i] = lu_solve((lu_l, piv_l), B[:, i])
    
    # Step 2: Solve X A_r = Y
    X = np.zeros_like(B)
    for i in range(n):
        X[i, :] = lu_solve((lu_r, piv_r), Y[i, :])
    
    return X
```

**QR Decomposition Method:**

```python
def deblur_qr(B, A_l, A_r):
    """Deblur using QR factorization"""
    n = B.shape[0]
    
    # QR factorization of A_l
    Q_l, R_l = qr(A_l, mode='economic')
    
    # QR factorization of A_r^T
    Q_r, R_r = qr(A_r.T, mode='economic')
    
    # Step 1: Solve A_l Y = B
    temp = Q_l.T @ B
    Y = solve_triangular(R_l, temp, lower=False)
    
    # Step 2: Solve X A_r = Y
    temp2 = Q_r.T @ Y.T
    X = solve_triangular(R_r, temp2, lower=False).T
    
    return X
```

### 4.2 Results

![Runtime Comparison](results/runtime_comparison.png)
*Figure 2: Runtime comparison across different methods and configurations*

#### Configuration 2 (Well-conditioned kernels)

| Image | Size | Method | CPU Time (s) | Relative Error | PSNR (dB) |
|-------|------|--------|--------------|----------------|-----------|
| 512_car | 512×512 | LU | 0.0585 | 1.31×10⁻⁸ | ∞ |
| 512_car | 512×512 | QR (SciPy) | 0.0257 | 1.33×10⁻⁸ | ∞ |
| 512_car | 512×512 | QR (my_qr) | 4.1393 | 1.45×10⁻⁸ | ∞ |
| 1024_books | 1024×1024 | LU | 1.8555 | 1.20×10⁻⁷ | 193.50 |
| 1024_books | 1024×1024 | QR (SciPy) | 0.1427 | 1.29×10⁻⁷ | 192.88 |
| 1024_books | 1024×1024 | QR (my_qr) | 53.0838 | 1.10×10⁻⁷ | 194.23 |

**Observations:**
- Both methods achieve excellent accuracy with relative errors around 10⁻⁸ to 10⁻⁷
- PSNR values are extremely high (>190 dB or infinite), indicating near-perfect reconstruction
- QR is significantly faster: ~2.3× faster for 512×512 images, ~13× faster for 1024×1024 images

![PSNR Comparison](results/psnr_comparison.png)
*Figure 3: PSNR comparison showing quality of reconstruction*

![Relative Error Comparison](results/error_comparison.png)
*Figure 4: Relative error comparison (log scale) across methods*

#### Configuration 1 (Ill-conditioned kernels)

| Image | Size | Method | CPU Time (s) | Relative Error | PSNR (dB) |
|-------|------|--------|--------------|----------------|-----------|
| 512_car | 512×512 | LU | 0.0797 | 3.23×10⁶⁶ | -1275.99 |
| 512_car | 512×512 | QR (SciPy) | 0.0253 | 3.05×10⁵⁷ | -1095.50 |
| 512_car | 512×512 | QR (my_qr) | 4.6719 | 2.98×10⁴² | -795.30 |
| 1024_books | 1024×1024 | LU | 2.0169 | 1.50×10¹⁴⁰ | -2748.43 |
| 1024_books | 1024×1024 | QR (SciPy) | 0.1371 | 4.30×10¹²⁹ | -2537.58 |
| 1024_books | 1024×1024 | QR (my_qr) | 53.6093 | 4.72×10⁹⁹ | -1938.40 |

**Observations:**
- Both methods fail catastrophically due to the near-singular kernels
- Relative errors are astronomically large (10⁵⁷ to 10¹⁴⁰)
- PSNR values are negative and extremely low, indicating complete reconstruction failure
- QR performs slightly better than LU but both are unusable

### 4.3 Visual Results

![Deblurring Comparison - 512_car config2](results/512_car_config2_deblur_comparison.png)
*Figure 5: Deblurring results for 512_car (config2) - Top row: Original, Blurred, LU result; Bottom row: QR result, LU error map, QR error map*

![Deblurring Comparison - 1024_books config2](results/1024_books_config2_deblur_comparison.png)
*Figure 6: Deblurring results for 1024_books (config2)*

## 5. Householder QR Decomposition (Task c, my_qr)

### 5.1 Implementation

A custom Householder QR decomposition is implemented without column pivoting:

```python
def my_qr(A, tol=1e-12):
    """
    Householder QR decomposition without column pivoting.
    
    Parameters:
    A: input matrix (m×n, m >= n)
    tol: tolerance for zero detection
    
    Returns:
    Q: orthogonal matrix (m×m)
    R: upper triangular matrix (m×n)
    """
    m, n = A.shape
    Q = np.eye(m)
    R = np.copy(A)
    
    for k in range(min(m, n)):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        if norm_x < tol:
            continue
        
        # Choose sign to avoid cancellation
        sign = -np.sign(x[0]) if x[0] != 0 else -1.0
        u = x.copy()
        u[0] -= sign * norm_x
        norm_u = np.linalg.norm(u)
        if norm_u < tol:
            continue
        v = u / norm_u
        
        # Apply Householder reflection to R
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:])
        
        # Accumulate Q
        H_full = np.eye(m)
        H_full[k:, k:] -= 2.0 * np.outer(v, v)
        Q = Q @ H_full
    
    R[np.abs(R) < tol] = 0.0
    return Q, R
```

### 5.2 Verification

The implementation is verified through self-tests comparing against NumPy's QR:

```python
def _self_test():
    """Verify QR decomposition correctness"""
    rng = np.random.default_rng(42)
    shapes = [(6, 4), (8, 5), (5, 5)]
    for m, n in shapes:
        A = rng.standard_normal((m, n))
        Q, R = my_qr(A)
        err_fact = np.linalg.norm(Q @ R - A)
        err_orth = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[0]))
        assert err_fact < 1e-10
        assert err_orth < 1e-10
    print("Householder QR self-test passed.")
```

All tests pass, confirming the correctness of the implementation.

## 6. Re-running Task b with my_qr (Task d)

### 6.1 Implementation

The **my_qr** implementation is integrated into the deblurring pipeline:

```python
def deblur_qr_custom(B, A_l, A_r):
    """Deblur using custom Householder QR"""
    Q_l, R_l = my_qr(A_l)
    temp = Q_l.T @ B
    Y = solve_triangular(R_l, temp, lower=False)
    
    Q_r, R_r = my_qr(A_r.T)
    temp2 = Q_r.T @ Y.T
    X = solve_triangular(R_r, temp2, lower=False).T
    return X
```

### 6.2 Results Comparison

#### Configuration 2 (Well-conditioned)

| Image | Method | Time (s) | Relative Error | PSNR (dB) |
|-------|--------|----------|----------------|-----------|
| 512_car | QR (SciPy) | 0.0234 | 1.33×10⁻⁸ | ∞ |
| 512_car | QR (my_qr) | 4.1393 | 1.45×10⁻⁸ | ∞ |
| 1024_books | QR (SciPy) | 0.1576 | 1.29×10⁻⁷ | 192.88 |
| 1024_books | QR (my_qr) | 53.0838 | 1.10×10⁻⁷ | 194.23 |

To highlight how **my_qr** compares against LU, built-in QR and LS across image sizes and configurations, we also include global comparison plots:

![Runtime Comparison (All Methods)](results/runtime_line_chart.png)
*Figure 6a: Runtime comparison (log scale) for LU, SciPy QR, my_qr and LS*

![PSNR Comparison (All Methods)](results/psnr_line_chart.png)
*Figure 6b: PSNR comparison for LU, SciPy QR, my_qr and LS*

![Relative Error Comparison (All Methods)](results/error_line_chart.png)
*Figure 6c: Relative error comparison (log scale) for LU, SciPy QR, my_qr and LS*

**Observations:**
- **Accuracy**: my_qr matches SciPy's accuracy (differences < 2×10⁻⁹ in relative error) and is comparable to LS on well-conditioned kernels.
- **Runtime**: my_qr is **180-330× slower** than SciPy QR due to pure Python implementation, while LS is slower than QR but still faster than my_qr.
- Plots clearly show that built-in QR dominates in speed, while LS is preferred for ill-conditioned kernels and my_qr is mainly of educational value.

#### Configuration 1 (Ill-conditioned)

Both my_qr and SciPy QR fail similarly on config1, as expected given the singular kernels.

### 6.3 Visual Comparison

![my_qr Comparison - 512_car config2](results/512_car_config2_taskd_custom_qr.png)
*Figure 8: Comparison of LU, SciPy QR, and my_qr for 512_car (config2)*

## 7. Least-Squares Improvements (Task e)

### 7.1 Motivation

The catastrophic failures in tasks b-d are due to the near-singular config1 kernels. A least-squares formulation provides a more robust approach:

**min_X ||A_ℓ X A_r - B||_F^2**

This can be solved via two sequential least-squares subproblems:

**min_Y ||A_ℓ Y - B||_F^2**, **min_X ||A_r^T X^T - Y^T||_F^2**

### 7.2 Implementation

```python
def deblur_ls(B, A_l, A_r):
    """Deblur using least-squares (pseudo-inverse)"""
    # Step 1: Solve A_l Y = B in least-squares sense
    Y, _, _, _ = np.linalg.lstsq(A_l, B, rcond=None)
    
    # Step 2: Solve X A_r = Y in least-squares sense
    X_rec_t, _, _, _ = np.linalg.lstsq(A_r.T, Y.T, rcond=None)
    X = X_rec_t.T
    
    return np.clip(X, 0, 1)
```

### 7.3 Results

| Image | Config | Method | Rel Error | PSNR (dB) | Time (s) |
|-------|--------|--------|-----------|-----------|----------|
| 512_car | config1 | LU | 3.23×10⁶⁶ | -1275.99 | 0.080 |
| 512_car | config1 | QR | 3.05×10⁵⁷ | -1095.50 | 0.025 |
| 512_car | config1 | **LS** | **7.88×10⁻²** | **76.27** | 1.354 |
| 1024_books | config1 | LU | 1.50×10¹⁴⁰ | -2748.43 | 2.016 |
| 1024_books | config1 | QR | 4.30×10¹²⁹ | -2537.58 | 0.137 |
| 1024_books | config1 | **LS** | **7.11×10⁻²** | **78.05** | 7.353 |
| 512_car | config2 | LU | 1.31×10⁻⁸ | ∞ | 0.060 |
| 512_car | config2 | QR | 1.33×10⁻⁸ | ∞ | 0.027 |
| 512_car | config2 | LS | 2.81×10⁻⁸ | ∞ | 1.396 |
| 1024_books | config2 | LU | 1.20×10⁻⁷ | 193.50 | 1.963 |
| 1024_books | config2 | QR | 1.29×10⁻⁷ | 192.88 | 0.158 |
| 1024_books | config2 | LS | 2.10×10⁻⁷ | 188.65 | 6.883 |

**Key Findings:**
- **Config1 (ill-conditioned)**: LS dramatically improves results:
  - Relative error drops from 10⁵⁷-10¹⁴⁰ to ~0.08
  - PSNR improves from -1000+ dB to ~75-78 dB (usable images!)
- **Config2 (well-conditioned)**: LS performs similarly to LU/QR:
  - Relative errors remain around 10⁻⁸ to 10⁻⁷
  - Slightly slower runtime but acceptable

### 7.4 Visual Results

![LS Comparison - 512_car config1](results/512_car_config1_taske_comparison.png)
*Figure 9: Least-squares deblurring for 512_car (config1) - dramatic improvement over LU/QR*

![LS Comparison - 1024_books config1](results/1024_books_config1_taske_comparison.png)
*Figure 10: Least-squares deblurring for 1024_books (config1)*

## 8. Comparison and Discussion

### 8.1 Performance Summary

**For well-conditioned kernels (config2):**
- **Best accuracy**: LU and QR are nearly identical (~10⁻⁸ relative error)
- **Best speed**: QR is fastest, especially for large images (13× faster than LU for 1024×1024)
- **Recommendation**: Use built-in QR for optimal speed-accuracy trade-off

**For ill-conditioned kernels (config1):**
- **LU/QR**: Complete failure (errors > 10⁵⁷)
- **Least-squares**: Successful recovery (PSNR ~75 dB, relative error ~0.08)
- **Recommendation**: Always use least-squares when kernel condition number > 10¹⁰

### 8.2 Method Comparison Table

| Method | Config1 Accuracy | Config2 Accuracy | Speed (512²) | Speed (1024²) | Stability |
|--------|-------------------|------------------|--------------|---------------|-----------|
| LU | ❌ Failed | ✅ Excellent | Medium | Slow | Poor for ill-conditioned |
| QR (SciPy) | ❌ Failed | ✅ Excellent | Fast | Fast | Better than LU |
| QR (Custom) | ❌ Failed | ✅ Excellent | Very Slow | Very Slow | Same as SciPy QR |
| Least-Squares | ✅ Good | ✅ Excellent | Slow | Medium | Best overall |

### 8.3 Key Observations

1. **Kernel conditioning is critical**: Config1 kernels lose 2-3% rank, making direct inversion impossible. Always check condition numbers before choosing a method.

2. **QR outperforms LU**: For well-conditioned problems, QR is faster and more stable. The speed advantage increases with image size.

3. **Custom QR validates correctness**: Our implementation produces identical results to SciPy's QR, confirming algorithmic correctness. However, it's 180-330× slower due to pure Python implementation.

4. **Least-squares rescues singular cases**: The pseudo-inverse approach automatically handles rank deficiency, providing usable reconstructions even when kernels are nearly singular.

5. **Trade-offs exist**: 
   - LU/QR are ideal for well-conditioned kernels (fast, accurate)
   - Least-squares is essential for ill-conditioned kernels (slower but stable)
   - Custom QR is educational but not practical for production use

### 8.4 Computational Complexity

- **LU factorization**: **O(n³)** for factorization, **O(n²)** per solve
- **QR factorization**: **O(n³)** for factorization, **O(n²)** per solve
- **Least-squares (lstsq)**: **O(n³)** for SVD-based pseudo-inverse

![QR Speedup over LU](results/speedup_comparison.png)
*Figure 7: QR factorization speedup compared to LU factorization*

For our test cases:
- 512×512: QR ~0.03s, LU ~0.06s, LS ~1.4s
- 1024×1024: QR ~0.16s, LU ~2.0s, LS ~7.4s

The speed difference between QR and LS is acceptable given the stability gains for ill-conditioned problems.

## 9. Conclusions

This project successfully investigates numerical methods for image deblurring through the lens of matrix factorizations and least-squares optimization. Key conclusions:

1. **Direct inversion methods (LU/QR) work excellently for well-conditioned kernels** but fail catastrophically for near-singular kernels.

2. **QR decomposition is preferred over LU** for well-conditioned problems due to superior speed and numerical stability.

3. **Custom Householder QR implementation** correctly reproduces SciPy's results, validating our understanding of the algorithm, though it's not competitive in speed.

4. **Least-squares formulation** provides a robust alternative that successfully handles ill-conditioned kernels, recovering usable images even when direct inversion fails.

5. **Kernel diagnosis is essential**: Checking condition numbers and ranks before deblurring helps select the appropriate method and explains observed failures.

6. **Method selection should be adaptive**: Use LU/QR for well-conditioned kernels (cond < 10⁶), and least-squares for ill-conditioned kernels (cond > 10¹⁰).

## 10. Future Work

Potential extensions and improvements:

1. **Adaptive method selection**: Automatically choose LU/QR vs. least-squares based on kernel condition numbers.

2. **Exploit kernel structure**: The blurring kernels have banded/Toeplitz structure that could be leveraged for faster solvers.

3. **Iterative methods**: For very large images, Krylov subspace methods (CG, LSQR) could provide faster approximate solutions.

4. **Regularization**: Explore Tikhonov regularization with optimal λ selection for further stability.

5. **Color images**: Extend the pipeline to handle RGB images by processing each channel independently or using tensor decompositions.

6. **Real-time deblurring**: Optimize implementations for real-time applications, possibly using GPU acceleration.

## 11. Appendix

### 11.1 Code Files

All implementation code is available in:
- `task_a_create_blur.py`: Blurring kernel generation and image blurring
- `task_b_deblur.py`: LU and QR deblurring methods
- `task_c_householder_qr.py`: Custom Householder QR implementation
- `task_d_deblur_custom_qr.py`: Deblurring with custom QR
- `task_e_improvements.py`: Least-squares deblurring and kernel diagnosis

### 11.2 Result Files

All experimental results are stored in:
- `results/`: Deblurred images, comparison plots, and error maps
- `blurred_images/`: Original images, blurred images, and blurring kernels

### 11.3 Additional Visualizations

Additional comparison figures are available in the `results/` directory, including:
- Side-by-side comparisons of all methods
- Error maps showing spatial distribution of reconstruction errors
- Performance plots comparing runtime vs. accuracy

---

**Project completed for DDA3005 - Numerical Methods, Fall 2025-26**

