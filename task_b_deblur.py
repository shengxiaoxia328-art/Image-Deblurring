"""
Task b: Implement deblurring algorithms using LU and QR factorizations
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve, qr, solve_triangular
import time
import matplotlib.pyplot as plt
import os

def deblur_lu(B, A_l, A_r):
    """
    Deblur image using LU factorization
    X = A_ℓ^(-1) B A_r^(-1)
    
    Using LU: A_ℓ = P_ℓ L_ℓ U_ℓ, so A_ℓ^(-1) = U_ℓ^(-1) L_ℓ^(-1) P_ℓ^T
    
    To solve A_ℓ X A_r = B for X:
    1. First solve A_ℓ Y = B, giving Y = A_ℓ^(-1) B
    2. Then solve Y A_r = B, which is equivalent to A_r^T X^T = Y^T
       So we solve A_r^T X^T = Y^T, giving X^T = (A_r^T)^(-1) Y^T
    
    Parameters:
    B: blurred image (n×n)
    A_l: left blurring kernel (n×n)
    A_r: right blurring kernel (n×n)
    
    Returns:
    X: deblurred image (n×n)
    """
    n = B.shape[0]
    
    # LU factorization of A_l
    lu_l, piv_l = lu_factor(A_l)
    
    # LU factorization of A_r^T (transpose for solving X A_r = Y)
    lu_r, piv_r = lu_factor(A_r.T)
    
    # Step 1: Solve A_l Y = B (for each column of B)
    Y = np.zeros_like(B)
    for i in range(n):
        Y[:, i] = lu_solve((lu_l, piv_l), B[:, i])
    
    # Step 2: Solve X A_r = Y, which is equivalent to A_r^T X^T = Y^T
    # So we solve for each column of X^T (which is each row of X)
    X = np.zeros_like(B)
    for i in range(n):
        X[i, :] = lu_solve((lu_r, piv_r), Y[i, :])
    
    return X

def deblur_qr(B, A_l, A_r):
    """
    Deblur image using QR factorization
    X = A_ℓ^(-1) B A_r^(-1)
    
    Using QR: A_ℓ = Q_ℓ R_ℓ, so A_ℓ^(-1) = R_ℓ^(-1) Q_ℓ^T
    
    To solve A_ℓ X A_r = B for X:
    1. First solve A_ℓ Y = B, giving Y = A_ℓ^(-1) B
    2. Then solve Y A_r = B, which is equivalent to A_r^T X^T = Y^T
    
    Parameters:
    B: blurred image (n×n)
    A_l: left blurring kernel (n×n)
    A_r: right blurring kernel (n×n)
    
    Returns:
    X: deblurred image (n×n)
    """
    n = B.shape[0]
    
    # QR factorization of A_l
    Q_l, R_l = qr(A_l, mode='economic')
    
    # QR factorization of A_r^T (transpose for solving X A_r = Y)
    Q_r, R_r = qr(A_r.T, mode='economic')
    
    # Step 1: Solve A_l Y = B
    # Compute Q_l^T B
    temp = Q_l.T @ B
    
    # Solve R_l Y = Q_l^T B (for each column)
    # Handle near-singular matrices by using lstsq if needed
    try:
        Y = solve_triangular(R_l, temp, lower=False)
    except:
        # Fallback to least squares if triangular solve fails
        Y = np.linalg.lstsq(R_l, temp, rcond=None)[0]
    
    # Step 2: Solve X A_r = Y, which is equivalent to A_r^T X^T = Y^T
    # So we solve for each row of X
    # First compute Q_r^T Y^T
    temp2 = Q_r.T @ Y.T
    
    # Solve R_r X^T = Q_r^T Y^T
    try:
        X = solve_triangular(R_r, temp2, lower=False).T
    except:
        # Fallback to least squares if triangular solve fails
        X = np.linalg.lstsq(R_r, temp2, rcond=None)[0].T
    
    return X

def compute_relative_error(X_rec, X_true):
    """
    Compute relative forward error using Frobenius norm
    ||X_rec - X_true||_F / ||X_true||_F
    """
    numerator = np.linalg.norm(X_rec - X_true, 'fro')
    denominator = np.linalg.norm(X_true, 'fro')
    if denominator < 1e-10:
        return float('inf')
    return numerator / denominator

def compute_psnr(X_rec, X_true):
    """
    Compute Peak Signal-to-Noise Ratio
    PSNR = 10 * log10(255^2 * n^2 / ||X_rec - X_true||_F^2)
    """
    mse = np.linalg.norm(X_rec - X_true, 'fro')**2
    n = X_rec.shape[0]
    if mse < 1e-10:
        return float('inf')
    psnr = 10 * np.log10(255**2 * n**2 / mse)
    return psnr

def main():
    # Create output directory
    os.makedirs('results', exist_ok=True)
    
    # Load blurred images and kernels from task a
    test_cases = [
        ('512_car', 'config1'),
        ('512_car', 'config2'),
        ('1024_books', 'config1'),
        ('1024_books', 'config2')
    ]
    
    results = []
    
    for img_name, config in test_cases:
        print(f"\n{'='*60}")
        print(f"Processing {img_name} with {config}")
        print(f"{'='*60}")
        
        # Load data
        X_true = np.load(f'blurred_images/{img_name}_original.npy')
        B = np.load(f'blurred_images/{img_name}_blurred_{config}.npy')
        A_l = np.load(f'blurred_images/{img_name}_A_l{config[-1]}.npy')
        A_r = np.load(f'blurred_images/{img_name}_A_r{config[-1]}.npy')
        
        n = X_true.shape[0]
        print(f"Image size: {n}×{n}")
        
        # Method 1: LU factorization
        print("\n--- LU Factorization Method ---")
        start_time = time.time()
        X_lu = deblur_lu(B, A_l, A_r)
        lu_time = time.time() - start_time
        
        lu_error = compute_relative_error(X_lu, X_true)
        lu_psnr = compute_psnr(X_lu, X_true)
        
        print(f"CPU Time: {lu_time:.4f} seconds")
        print(f"Relative Error: {lu_error:.6e}")
        print(f"PSNR: {lu_psnr:.2f} dB")
        
        # Method 2: QR factorization
        print("\n--- QR Factorization Method ---")
        start_time = time.time()
        X_qr = deblur_qr(B, A_l, A_r)
        qr_time = time.time() - start_time
        
        qr_error = compute_relative_error(X_qr, X_true)
        qr_psnr = compute_psnr(X_qr, X_true)
        
        print(f"CPU Time: {qr_time:.4f} seconds")
        print(f"Relative Error: {qr_error:.6e}")
        print(f"PSNR: {qr_psnr:.2f} dB")
        
        # Save results
        results.append({
            'image': img_name,
            'config': config,
            'size': n,
            'lu_time': lu_time,
            'lu_error': lu_error,
            'lu_psnr': lu_psnr,
            'qr_time': qr_time,
            'qr_error': qr_error,
            'qr_psnr': qr_psnr
        })
        
        # Visualize results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Original, Blurred, LU result
        axes[0, 0].imshow(X_true, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(B, cmap='gray')
        axes[0, 1].set_title('Blurred Image')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(X_lu, cmap='gray')
        axes[0, 2].set_title(f'LU Deblurred\n(Error: {lu_error:.2e}, PSNR: {lu_psnr:.2f} dB)')
        axes[0, 2].axis('off')
        
        # Row 2: QR result, Difference images
        axes[1, 0].imshow(X_qr, cmap='gray')
        axes[1, 0].set_title(f'QR Deblurred\n(Error: {qr_error:.2e}, PSNR: {qr_psnr:.2f} dB)')
        axes[1, 0].axis('off')
        
        # Difference images (scaled for visualization)
        diff_lu = np.abs(X_lu - X_true)
        diff_lu_scaled = diff_lu / (np.max(diff_lu) + 1e-10)
        axes[1, 1].imshow(diff_lu_scaled, cmap='hot')
        axes[1, 1].set_title('LU Error Map')
        axes[1, 1].axis('off')
        
        diff_qr = np.abs(X_qr - X_true)
        diff_qr_scaled = diff_qr / (np.max(diff_qr) + 1e-10)
        axes[1, 2].imshow(diff_qr_scaled, cmap='hot')
        axes[1, 2].set_title('QR Error Map')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/{img_name}_{config}_deblur_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save deblurred images
        np.save(f'results/{img_name}_{config}_deblurred_lu.npy', X_lu)
        np.save(f'results/{img_name}_{config}_deblurred_qr.npy', X_qr)
        plt.imsave(f'results/{img_name}_{config}_deblurred_lu.png', X_lu, cmap='gray')
        plt.imsave(f'results/{img_name}_{config}_deblurred_qr.png', X_qr, cmap='gray')
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Image':<15} {'Config':<10} {'Size':<8} {'Method':<5} {'Time (s)':<12} {'Rel Error':<15} {'PSNR (dB)':<12}")
    print("-"*80)
    
    for r in results:
        print(f"{r['image']:<15} {r['config']:<10} {r['size']:<8} {'LU':<5} {r['lu_time']:<12.4f} {r['lu_error']:<15.6e} {r['lu_psnr']:<12.2f}")
        print(f"{'':<15} {'':<10} {'':<8} {'QR':<5} {r['qr_time']:<12.4f} {r['qr_error']:<15.6e} {r['qr_psnr']:<12.2f}")
        print("-"*80)
    
    # Observations
    print("\n" + "="*80)
    print("OBSERVATIONS")
    print("="*80)
    print("1. CPU Time Comparison:")
    for r in results:
        speedup = r['lu_time'] / r['qr_time'] if r['qr_time'] > 0 else 0
        print(f"   {r['image']} ({r['config']}): LU is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than QR")
    
    print("\n2. Accuracy Comparison:")
    for r in results:
        better = "LU" if r['lu_error'] < r['qr_error'] else "QR"
        print(f"   {r['image']} ({r['config']}): {better} has lower relative error")
    
    print("\n3. PSNR Comparison:")
    for r in results:
        better = "LU" if r['lu_psnr'] > r['qr_psnr'] else "QR"
        print(f"   {r['image']} ({r['config']}): {better} has higher PSNR (better quality)")
    
    print("\nTask b completed! Results saved in 'results' directory.")

if __name__ == '__main__':
    main()

