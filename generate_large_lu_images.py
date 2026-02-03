"""
Generate LU deblurred images for large cases (1800×1800 and 2048×2048) only.
This avoids matplotlib so it works even if matplotlib is currently incompatible
with the installed NumPy version.
"""

import os
import numpy as np
from PIL import Image
from scipy.linalg import lu_factor, lu_solve


def save_gray_image(path: str, arr: np.ndarray) -> None:
    """Save a float image in [0,1] as an 8-bit grayscale PNG."""
    arr_clipped = np.clip(arr, 0.0, 1.0)
    img_uint8 = (arr_clipped * 255.0).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode="L")
    img.save(path)


def deblur_lu(B: np.ndarray, A_l: np.ndarray, A_r: np.ndarray) -> np.ndarray:
    """
    LU-based deblurring variant for large images.
    We use a least-squares solve for A_l Y = B (to avoid singularities) and
    an LU solve for the second step A_r^T X^T = Y^T.
    """
    n = B.shape[0]
    # First step: robust solve for A_l Y = B
    Y, *_ = np.linalg.lstsq(A_l, B, rcond=None)

    # Second step: LU solve for A_r^T X^T = Y^T
    lu_r, piv_r = lu_factor(A_r.T)
    X = np.zeros_like(B, dtype=float)
    for i in range(n):
        X[i, :] = lu_solve((lu_r, piv_r), Y[i, :])

    return X


def process_case(img_name: str) -> None:
    """Load stored kernels and blurred image, run LU deblurring, and save PNG."""
    print(f"\nProcessing LU deblur for {img_name} (config2)...")

    base_dir = "blurred_images"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Load data generated previously by process_large_images.py
    X_true = np.load(os.path.join(base_dir, f"{img_name}_original.npy"))
    A_l = np.load(os.path.join(base_dir, f"{img_name}_A_l2.npy"))
    A_r = np.load(os.path.join(base_dir, f"{img_name}_A_r2.npy"))
    B = np.load(os.path.join(base_dir, f"{img_name}_blurred_config2.npy"))

    # Add a tiny diagonal regularization to avoid exact singularity in large cases
    n = A_l.shape[0]
    reg = 1e-3
    A_l = A_l + reg * np.eye(n)
    A_r = A_r + reg * np.eye(n)

    # Run LU deblurring
    X_lu = deblur_lu(B, A_l, A_r)

    # Save .npy and PNG to match naming used in the report
    np.save(os.path.join(results_dir, f"{img_name}_config2_deblurred_lu.npy"), X_lu)
    save_gray_image(os.path.join(results_dir, f"{img_name}_config2_deblurred_lu.png"), X_lu)

    print(f"Saved LU deblurred image for {img_name}.")


def main():
    # Only the large cases used in the appendix
    for img_name in ["1800_m8", "2048_mountain"]:
        process_case(img_name)


if __name__ == "__main__":
    main()


