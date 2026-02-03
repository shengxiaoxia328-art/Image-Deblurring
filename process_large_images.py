"""
Process 2048 and 4096 images for line charts and appendix
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import sys

# Import functions from existing scripts
sys.path.insert(0, '.')
from task_a_create_blur import create_blurring_kernel, create_symmetric_banded_kernel, blur_image, load_image
from task_b_deblur import deblur_lu, deblur_qr, compute_relative_error, compute_psnr
from task_d_deblur_custom_qr import deblur_qr_custom
from task_e_improvements import deblur_least_squares

def process_image(img_name, img_size):
    """Process a single image: blur and deblur with all four methods"""
    print(f"\n{'='*80}")
    print(f"Processing {img_name} ({img_size}×{img_size})")
    print(f"{'='*80}")
    
    os.makedirs('blurred_images', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load original image
    img_path = f'test_images/{img_name}_original.png'
    X_true = load_image(img_path)
    
    # Resize if needed
    if X_true.shape[0] != img_size or X_true.shape[1] != img_size:
        X_true = np.array(Image.fromarray(X_true).resize((img_size, img_size), Image.LANCZOS))
    
    # Save original
    np.save(f'blurred_images/{img_name}_original.npy', X_true)
    
    # Create blurring kernels (config2 - well-conditioned)
    n = img_size
    A_l = create_blurring_kernel(n, j=5, k=3)
    A_r = create_symmetric_banded_kernel(n, num_bands=5)

    # For very large images, add a tiny diagonal term to avoid exact singularity
    # (used only for scalability experiments in the appendix, not for main tasks)
    if n >= 1800:
        reg = 1e-6
        A_l = A_l + reg * np.eye(n)
        A_r = A_r + reg * np.eye(n)
    
    # Normalize rows to sum to 1 for stability
    row_sums_l = A_l.sum(axis=1, keepdims=True)
    row_sums_l[row_sums_l == 0] = 1
    A_l = A_l / row_sums_l
    
    row_sums_r = A_r.sum(axis=1, keepdims=True)
    row_sums_r[row_sums_r == 0] = 1
    A_r = A_r / row_sums_r
    
    # Save kernels
    np.save(f'blurred_images/{img_name}_A_l2.npy', A_l)
    np.save(f'blurred_images/{img_name}_A_r2.npy', A_r)
    
    # Generate blurred image
    B = blur_image(X_true, A_l, A_r)
    np.save(f'blurred_images/{img_name}_blurred_config2.npy', B)
    
    print("Blurred image created")
    
    # Deblur with all four methods
    results = {}
    
    # Method 1: LU
    print("\n--- LU Method ---")
    start = time.time()
    try:
        X_lu = deblur_lu(B, A_l, A_r)
        lu_time = time.time() - start
        lu_error = compute_relative_error(X_lu, X_true)
        lu_psnr = compute_psnr(X_lu, X_true)
        results['LU'] = {'X': X_lu, 'time': lu_time, 'error': lu_error, 'psnr': lu_psnr}
        print(f"  Time: {lu_time:.4f}s, Error: {lu_error:.3e}, PSNR: {lu_psnr:.2f} dB")
    except Exception as e:
        print(f"  Failed: {e}")
        results['LU'] = None
    
    # Method 2: QR (SciPy)
    print("\n--- QR (SciPy) Method ---")
    start = time.time()
    try:
        X_qr = deblur_qr(B, A_l, A_r)
        qr_time = time.time() - start
        qr_error = compute_relative_error(X_qr, X_true)
        qr_psnr = compute_psnr(X_qr, X_true)
        results['QR'] = {'X': X_qr, 'time': qr_time, 'error': qr_error, 'psnr': qr_psnr}
        print(f"  Time: {qr_time:.4f}s, Error: {qr_error:.3e}, PSNR: {qr_psnr:.2f} dB")
    except Exception as e:
        print(f"  Failed: {e}")
        results['QR'] = None
    
    # Method 3: my_qr (skip for large images > 1024 to save time)
    if n <= 1024:
        print("\n--- QR (my_qr) Method ---")
        start = time.time()
        try:
            X_myqr = deblur_qr_custom(B, A_l, A_r)
            myqr_time = time.time() - start
            myqr_error = compute_relative_error(X_myqr, X_true)
            myqr_psnr = compute_psnr(X_myqr, X_true)
            results['my_qr'] = {'X': X_myqr, 'time': myqr_time, 'error': myqr_error, 'psnr': myqr_psnr}
            print(f"  Time: {myqr_time:.4f}s, Error: {myqr_error:.3e}, PSNR: {myqr_psnr:.2f} dB")
        except Exception as e:
            print(f"  Failed: {e}")
            results['my_qr'] = None
    else:
        print("\n--- QR (my_qr) Method ---")
        print("  Skipped for large images (too slow)")
        results['my_qr'] = None
    
    # Method 4: LS
    print("\n--- LS Method ---")
    start = time.time()
    try:
        X_ls = deblur_least_squares(B, A_l, A_r)
        ls_time = time.time() - start
        ls_error = compute_relative_error(X_ls, X_true)
        ls_psnr = compute_psnr(X_ls, X_true)
        results['LS'] = {'X': X_ls, 'time': ls_time, 'error': ls_error, 'psnr': ls_psnr}
        print(f"  Time: {ls_time:.4f}s, Error: {ls_error:.3e}, PSNR: {ls_psnr:.2f} dB")
    except Exception as e:
        print(f"  Failed: {e}")
        results['LS'] = None
    
    # Save deblurred images
    for method, data in results.items():
        if data is not None:
            method_key = {'LU': 'lu', 'QR': 'qr', 'my_qr': 'qr_custom', 'LS': 'ls'}[method]
            np.save(f'results/{img_name}_config2_deblurred_{method_key}.npy', data['X'])
            plt.imsave(f'results/{img_name}_config2_deblurred_{method_key}.png', 
                      np.clip(data['X'], 0, 1), cmap='gray')
    
    # Save blurred image
    plt.imsave(f'results/{img_name}_config2_blurred.png', np.clip(B, 0, 1), cmap='gray')
    
    # Save original
    plt.imsave(f'results/{img_name}_original.png', np.clip(X_true, 0, 1), cmap='gray')
    
    return results

def main():
    # Process 1800 and 2048 images
    test_cases = [
        ('1800_m8', 1800),
        ('2048_mountain', 2048),
    ]
    
    all_results = {}
    for img_name, img_size in test_cases:
        results = process_image(img_name, img_size)
        all_results[img_name] = results
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Image':<20} {'Method':<10} {'Time(s)':<12} {'Error':<15} {'PSNR(dB)':<12}")
    print("-"*80)
    
    # Save summary to file for line chart update
    summary_file = 'results/large_image_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("SUMMARY OF LARGE IMAGE PROCESSING\n")
        f.write("="*80 + "\n")
        f.write(f"{'Image':<20} {'Method':<10} {'Time(s)':<12} {'Error':<15} {'PSNR(dB)':<12}\n")
        f.write("-"*80 + "\n")
        
        for img_name, results in all_results.items():
            for method, data in results.items():
                if data is not None:
                    line = f"{img_name:<20} {method:<10} {data['time']:<12.4f} {data['error']:<15.3e} {data['psnr']:<12.2f}\n"
                    print(line.strip())
                    f.write(line)
    
    print(f"\nSummary saved to {summary_file}")
    print("\nAll images processed!")

if __name__ == '__main__':
    main()

