"""
Task f: Padding method for deblurring.
Padding extends image borders with white pixels to create smoother blurred images
and potentially improve deblurring quality.

This script is implemented WITHOUT importing matplotlib-dependent modules
so that it can run even when the Matplotlib/Numpy binary versions mismatch.
"""

import numpy as np
from scipy.linalg import qr, solve_triangular


def _deblur_qr_basic(B, A_l, A_r):
    """
    Minimal QR-based deblurring implementation (no matplotlib imports).
    Mirrors the logic of task_b_deblur.deblur_qr:
      1) Solve A_l Y = B
      2) Solve A_r^T X^T = Y^T
    """
    n = B.shape[0]

    # QR factorization of A_l
    Q_l, R_l = qr(A_l, mode="economic")

    # QR factorization of A_r^T
    Q_r, R_r = qr(A_r.T, mode="economic")

    # Step 1: A_l Y = B  ->  R_l Y = Q_l^T B
    temp = Q_l.T @ B
    try:
        Y = solve_triangular(R_l, temp, lower=False)
    except Exception:
        Y = np.linalg.lstsq(R_l, temp, rcond=None)[0]

    # Step 2: Y A_r = B  ->  A_r^T X^T = Y^T  ->  R_r X^T = Q_r^T Y^T
    temp2 = Q_r.T @ Y.T
    try:
        X = solve_triangular(R_r, temp2, lower=False).T
    except Exception:
        X = np.linalg.lstsq(R_r, temp2, rcond=None)[0].T

    return X


def compute_relative_error(X_rec, X_true):
    """Relative forward error in Frobenius norm."""
    num = np.linalg.norm(X_rec - X_true, ord="fro")
    den = np.linalg.norm(X_true, ord="fro") + 1e-15
    return num / den


def compute_psnr(X_rec, X_true, max_val=1.0):
    """Compute PSNR (dB) for images in [0, 1]."""
    mse = np.mean((X_rec - X_true) ** 2)
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(max_val**2 / mse)


def deblur_least_squares(B, A_l, A_r):
    """
    Simple least-squares deblurring (no external imports):
      1) Solve A_l Y ≈ B in least-squares sense
      2) Solve A_r^T X^T ≈ Y^T in least-squares sense
    """
    # Solve A_l Y ≈ B
    Y, *_ = np.linalg.lstsq(A_l, B, rcond=None)
    # Solve A_r^T X^T ≈ Y^T
    X_T, *_ = np.linalg.lstsq(A_r.T, Y.T, rcond=None)
    return X_T.T


def deblur_with_padding(B, A_l, A_r, pad_size=10):
    """
    Deblur image using padding method.
    
    The padding approach extends the image borders with white pixels (value=1.0)
    before applying deblurring. This can result in smoother/more natural blurred
    images and potentially improve reconstruction quality, especially near boundaries.
    
    Steps:
    1. Pad the blurred image B with white pixels (value=1.0)
    2. Extend kernels A_l and A_r accordingly
    3. Apply deblurring on the padded system
    4. Extract the original-sized image from the center
    
    Parameters:
    B: blurred image (n×n)
    A_l: left blurring kernel (n×n)
    A_r: right blurring kernel (n×n)
    pad_size: number of pixels to pad on each side (default: 10)
    
    Returns:
    X: deblurred image (n×n)
    """
    n = B.shape[0]
    
    # Pad blurred image with white pixels (value=1.0 for grayscale)
    B_padded = np.pad(B, pad_size, mode='constant', constant_values=1.0)
    n_padded = B_padded.shape[0]
    
    # Extend kernels by padding with identity-like structure
    # For A_l: pad with identity matrix (no blurring on padded region)
    A_l_padded = np.eye(n_padded)
    A_l_padded[pad_size:pad_size+n, pad_size:pad_size+n] = A_l
    
    # For A_r: pad with identity matrix
    A_r_padded = np.eye(n_padded)
    A_r_padded[pad_size:pad_size+n, pad_size:pad_size+n] = A_r
    
    # Normalize padded kernels to maintain row sums
    row_sums_l = A_l_padded.sum(axis=1, keepdims=True)
    row_sums_l[row_sums_l == 0] = 1
    A_l_padded = A_l_padded / row_sums_l
    
    row_sums_r = A_r_padded.sum(axis=1, keepdims=True)
    row_sums_r[row_sums_r == 0] = 1
    A_r_padded = A_r_padded / row_sums_r
    
    # Apply deblurring on padded system using minimal QR solver (most stable)
    try:
        X_padded = _deblur_qr_basic(B_padded, A_l_padded, A_r_padded)
    except Exception:
        # Fallback to least squares if QR fails
        X_padded = deblur_least_squares(B_padded, A_l_padded, A_r_padded)
    
    # Extract original-sized image from center
    X = X_padded[pad_size:pad_size+n, pad_size:pad_size+n]
    
    # Clip to valid range
    X = np.clip(X, 0, 1)
    
    return X


if __name__ == "__main__":
    import os
    import time
    from PIL import Image
    
    os.makedirs("results", exist_ok=True)
    
    # Include both small and large images so Appendix padding images exist
    test_cases = [
        ("512_car", "config1"),
        ("512_car", "config2"),
        ("1024_books", "config1"),
        ("1024_books", "config2"),
        ("1800_m8", "config2"),
        ("2048_mountain", "config2"),
    ]
    
    for img_name, config in test_cases:
        print(f"\nProcessing {img_name} ({config})...")
        
        X_true = np.load(f"blurred_images/{img_name}_original.npy")
        B = np.load(f"blurred_images/{img_name}_blurred_{config}.npy")
        A_l = np.load(f"blurred_images/{img_name}_A_l{config[-1]}.npy")
        A_r = np.load(f"blurred_images/{img_name}_A_r{config[-1]}.npy")
        
        start = time.time()
        X_pad = deblur_with_padding(B, A_l, A_r)
        pad_time = time.time() - start
        pad_error = compute_relative_error(X_pad, X_true)
        pad_psnr = compute_psnr(X_pad, X_true)
        
        print(f"Padding | t={pad_time:.3f}s err={pad_error:.3e} PSNR={pad_psnr:.2f} dB")
        
        np.save(f"results/{img_name}_{config}_deblurred_pad.npy", X_pad)
        # Save PNG using PIL (avoids matplotlib dependency)
        img_uint8 = (np.clip(X_pad, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img_uint8, mode="L").save(f"results/{img_name}_{config}_deblurred_pad.png")

