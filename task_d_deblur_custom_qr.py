"""
Task d: Repeat the deblurring experiments (task b) but replace the built-in QR
factorization with our own Householder implementation from task c. We keep the LU
baseline and also keep the SciPy QR results for direct comparison.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_triangular

from task_b_deblur import (
    deblur_lu,
    deblur_qr as deblur_qr_builtin,
    compute_relative_error,
    compute_psnr,
)
from task_c_householder_qr import my_qr


def _safe_upper_solve(R, rhs):
    """
    Solve R x = rhs for upper-triangular R. Fall back to least-squares if needed.
    """
    try:
        return solve_triangular(R, rhs, lower=False)
    except Exception:
        return np.linalg.lstsq(R, rhs, rcond=None)[0]


def deblur_qr_custom(B, A_l, A_r):
    """
    Deblur image using the custom Householder QR factorization.
    """
    Q_l, R_l = my_qr(A_l)
    temp = Q_l.T @ B
    Y = _safe_upper_solve(R_l, temp)

    Q_r, R_r = my_qr(A_r.T)
    temp2 = Q_r.T @ Y.T
    X = _safe_upper_solve(R_r, temp2).T
    return X


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
        print(f"\n{'='*80}")
        print(f"Processing {img_name} with {config}")
        print(f"{'='*80}")

        X_true = np.load(f"blurred_images/{img_name}_original.npy")
        B = np.load(f"blurred_images/{img_name}_blurred_{config}.npy")
        A_l = np.load(f"blurred_images/{img_name}_A_l{config[-1]}.npy")
        A_r = np.load(f"blurred_images/{img_name}_A_r{config[-1]}.npy")

        n = X_true.shape[0]
        print(f"Image size: {n}×{n}")

        # LU baseline
        start = time.time()
        X_lu = deblur_lu(B, A_l, A_r)
        lu_time = time.time() - start
        lu_error = compute_relative_error(X_lu, X_true)
        lu_psnr = compute_psnr(X_lu, X_true)

        # Built-in SciPy QR (reference)
        start = time.time()
        X_qr_builtin = deblur_qr_builtin(B, A_l, A_r)
        qr_builtin_time = time.time() - start
        qr_builtin_error = compute_relative_error(X_qr_builtin, X_true)
        qr_builtin_psnr = compute_psnr(X_qr_builtin, X_true)

        # Custom QR
        start = time.time()
        X_qr_custom = deblur_qr_custom(B, A_l, A_r)
        qr_custom_time = time.time() - start
        qr_custom_error = compute_relative_error(X_qr_custom, X_true)
        qr_custom_psnr = compute_psnr(X_qr_custom, X_true)

        print(f"LU         | time: {lu_time:.4f}s | err: {lu_error:.3e} | PSNR: {lu_psnr:.2f} dB")
        print(f"QR (SciPy) | time: {qr_builtin_time:.4f}s | err: {qr_builtin_error:.3e} | PSNR: {qr_builtin_psnr:.2f} dB")
        print(f"QR (ours)  | time: {qr_custom_time:.4f}s | err: {qr_custom_error:.3e} | PSNR: {qr_custom_psnr:.2f} dB")

        results.append(
            {
                "image": img_name,
                "config": config,
                "size": n,
                "lu_time": lu_time,
                "lu_error": lu_error,
                "lu_psnr": lu_psnr,
                "qr_builtin_time": qr_builtin_time,
                "qr_builtin_error": qr_builtin_error,
                "qr_builtin_psnr": qr_builtin_psnr,
                "qr_custom_time": qr_custom_time,
                "qr_custom_error": qr_custom_error,
                "qr_custom_psnr": qr_custom_psnr,
            }
        )

        fig, axes = plt.subplots(2, 4, figsize=(20, 10))

        axes[0, 0].imshow(X_true, cmap="gray")
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(B, cmap="gray")
        axes[0, 1].set_title("Blurred")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(X_lu, cmap="gray")
        axes[0, 2].set_title(f"LU\nerr={lu_error:.2e}\nPSNR={lu_psnr:.1f}")
        axes[0, 2].axis("off")

        axes[0, 3].imshow(X_qr_builtin, cmap="gray")
        axes[0, 3].set_title(f"QR (SciPy)\nerr={qr_builtin_error:.2e}\nPSNR={qr_builtin_psnr:.1f}")
        axes[0, 3].axis("off")

        axes[1, 0].imshow(X_qr_custom, cmap="gray")
        axes[1, 0].set_title(f"QR (ours)\nerr={qr_custom_error:.2e}\nPSNR={qr_custom_psnr:.1f}")
        axes[1, 0].axis("off")

        diff_builtin = np.abs(X_qr_builtin - X_true)
        axes[1, 1].imshow(diff_builtin / (diff_builtin.max() + 1e-10), cmap="hot")
        axes[1, 1].set_title("QR (SciPy) error")
        axes[1, 1].axis("off")

        diff_custom = np.abs(X_qr_custom - X_true)
        axes[1, 2].imshow(diff_custom / (diff_custom.max() + 1e-10), cmap="hot")
        axes[1, 2].set_title("QR (ours) error")
        axes[1, 2].axis("off")

        diff_between = np.abs(X_qr_custom - X_qr_builtin)
        axes[1, 3].imshow(diff_between / (diff_between.max() + 1e-10), cmap="hot")
        axes[1, 3].set_title("|ours - SciPy|")
        axes[1, 3].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"results/{img_name}_{config}_taskd_custom_qr.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        np.save(f"results/{img_name}_{config}_deblurred_qr_custom.npy", X_qr_custom)
        plt.imsave(f"results/{img_name}_{config}_deblurred_qr_custom.png", X_qr_custom, cmap="gray")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Image':<12} {'Config':<8} {'Method':<12} {'Time(s)':<10} {'Rel Error':<15} {'PSNR (dB)':<12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['image']:<12} {r['config']:<8} {'LU':<12} {r['lu_time']:<10.4f} {r['lu_error']:<15.3e} {r['lu_psnr']:<12.2f}")
        print(f"{'':<12} {'':<8} {'QR (SciPy)':<12} {r['qr_builtin_time']:<10.4f} {r['qr_builtin_error']:<15.3e} {r['qr_builtin_psnr']:<12.2f}")
        print(f"{'':<12} {'':<8} {'QR (ours)':<12} {r['qr_custom_time']:<10.4f} {r['qr_custom_error']:<15.3e} {r['qr_custom_psnr']:<12.2f}")
        print("-" * len(header))

    print("\nOBSERVATIONS")
    print("=" * 80)
    for r in results:
        err_delta = abs(r["qr_custom_error"] - r["qr_builtin_error"])
        psnr_delta = r["qr_custom_psnr"] - r["qr_builtin_psnr"]
        if not np.isfinite(psnr_delta):
            psnr_delta_str = "n/a"
        else:
            psnr_delta_str = f"{psnr_delta:.2f} dB"
        if r["qr_builtin_time"] > 0:
            time_ratio = r["qr_custom_time"] / r["qr_builtin_time"]
            time_ratio_str = f"{time_ratio:.2f}x"
        else:
            time_ratio_str = "∞"
        print(
            f"{r['image']} ({r['config']}): "
            f"|Δerr|={err_delta:.2e}, ΔPSNR={psnr_delta_str}, "
            f"our QR runtime = {time_ratio_str} SciPy QR."
        )

    print("\nTask d completed! Custom QR results saved in 'results/'.")


if __name__ == "__main__":
    main()

