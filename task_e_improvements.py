"""
Task e: Improve the deblurring pipeline by (1) diagnosing singular/ill-conditioned
blurring kernels and (2) applying a regularized least-squares formulation

    min_X ||A_l X A_r - B||_F^2 + λ ||X||_F^2

which is approximately solved via two sequential ridge regressions.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from task_b_deblur import (
    deblur_lu,
    deblur_qr,
    compute_relative_error,
    compute_psnr,
)


def analyze_kernel(A, name: str):
    cond = np.linalg.cond(A)
    rank = np.linalg.matrix_rank(A)
    print(f"{name}: cond={cond:.2e}, rank={rank}/{A.shape[0]}")
    return cond, rank


def deblur_least_squares(B, A_l, A_r):
    """Deblur using least-squares (pseudo-inverse)"""
    # Step 1: Solve A_l Y ≈ B in least-squares sense
    Y, *_ = np.linalg.lstsq(A_l, B, rcond=None)
    
    # Step 2: Solve X A_r ≈ Y (via A_r^T X^T ≈ Y^T)
    X_T, *_ = np.linalg.lstsq(A_r.T, Y.T, rcond=None)
    X = X_T.T
    
    return np.clip(X, 0, 1)


def main():
    os.makedirs("results", exist_ok=True)

    test_cases = [
        ("512_car", "config1"),
        ("512_car", "config2"),
        ("1024_books", "config1"),
        ("1024_books", "config2"),
    ]

    results = []

    for img_name, config in test_cases:
        print("\n" + "=" * 80)
        print(f"Processing {img_name} ({config})")
        print("=" * 80)

        X_true = np.load(f"blurred_images/{img_name}_original.npy")
        B = np.load(f"blurred_images/{img_name}_blurred_{config}.npy")
        A_l = np.load(f"blurred_images/{img_name}_A_l{config[-1]}.npy")
        A_r = np.load(f"blurred_images/{img_name}_A_r{config[-1]}.npy")

        cond_l, rank_l = analyze_kernel(A_l, "A_l")
        cond_r, rank_r = analyze_kernel(A_r, "A_r")
        print("Solving least-squares system without regularization.")

        start = time.time()
        X_lu = deblur_lu(B, A_l, A_r)
        lu_time = time.time() - start
        lu_error = compute_relative_error(X_lu, X_true)
        lu_psnr = compute_psnr(X_lu, X_true)

        start = time.time()
        X_qr = deblur_qr(B, A_l, A_r)
        qr_time = time.time() - start
        qr_error = compute_relative_error(X_qr, X_true)
        qr_psnr = compute_psnr(X_qr, X_true)

        start = time.time()
        X_ls = deblur_least_squares(B, A_l, A_r)
        ls_time = time.time() - start
        ls_error = compute_relative_error(X_ls, X_true)
        ls_psnr = compute_psnr(X_ls, X_true)

        print(f"LU    | t={lu_time:.3f}s err={lu_error:.3e} PSNR={lu_psnr:.2f} dB")
        print(f"QR    | t={qr_time:.3f}s err={qr_error:.3e} PSNR={qr_psnr:.2f} dB")
        print(f"LS    | t={ls_time:.3f}s err={ls_error:.3e} PSNR={ls_psnr:.2f} dB")

        results.append(
            {
                "image": img_name,
                "config": config,
                "lu_error": lu_error,
                "lu_psnr": lu_psnr,
                "qr_error": qr_error,
                "qr_psnr": qr_psnr,
                "ls_error": ls_error,
                "ls_psnr": ls_psnr,
                "lu_time": lu_time,
                "qr_time": qr_time,
                "ls_time": ls_time,
                "cond_l": cond_l,
                "cond_r": cond_r,
            }
        )

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes[0, 0].imshow(X_true, cmap="gray"); axes[0, 0].set_title("Original"); axes[0, 0].axis("off")
        axes[0, 1].imshow(B, cmap="gray"); axes[0, 1].set_title("Blurred"); axes[0, 1].axis("off")
        axes[0, 2].imshow(X_lu, cmap="gray"); axes[0, 2].set_title(f"LU\nerr={lu_error:.2e}")
        axes[0, 2].axis("off")
        axes[1, 0].imshow(X_qr, cmap="gray"); axes[1, 0].set_title(f"QR\nerr={qr_error:.2e}"); axes[1, 0].axis("off")
        axes[1, 1].imshow(X_ls, cmap="gray"); axes[1, 1].set_title(f"LS\nerr={ls_error:.2e}"); axes[1, 1].axis("off")
        diff_ls = np.abs(X_ls - X_true)
        axes[1, 2].imshow(diff_ls / (diff_ls.max() + 1e-10), cmap="hot"); axes[1, 2].set_title("LS error map")
        axes[1, 2].axis("off")
        plt.tight_layout()
        plt.savefig(f"results/{img_name}_{config}_taske_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

        np.save(f"results/{img_name}_{config}_deblurred_ls.npy", X_ls)
        plt.imsave(f"results/{img_name}_{config}_deblurred_ls.png", X_ls, cmap="gray")

    print("\nSUMMARY (Task e)")
    header = f"{'Image':<12}{'Config':<10}{'Method':<8}{'RelErr':<15}{'PSNR':<10}{'Time(s)':<10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['image']:<12}{r['config']:<10}{'LU':<8}{r['lu_error']:<15.3e}{r['lu_psnr']:<10.2f}{r['lu_time']:<10.3f}")
        print(f"{'':<12}{'':<10}{'QR':<8}{r['qr_error']:<15.3e}{r['qr_psnr']:<10.2f}{r['qr_time']:<10.3f}")
        print(f"{'':<12}{'':<10}{'LS':<8}{r['ls_error']:<15.3e}{r['ls_psnr']:<10.2f}{r['ls_time']:<10.3f}")
        print("-" * len(header))

    improvements = []
    for r in results:
        better = "LS" if r["ls_error"] < r["qr_error"] else "QR"
        improvements.append((r["image"], r["config"], better))

    print("\nOBSERVATIONS")
    for img, cfg, better in improvements:
        print(f"{img} ({cfg}): best accuracy -> {better}")


if __name__ == "__main__":
    main()

