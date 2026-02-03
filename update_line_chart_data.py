"""
Extract data from processing results and update line chart data
"""
import numpy as np
import os
import sys
from task_b_deblur import compute_relative_error, compute_psnr

def extract_results(img_name, config='config2'):
    """Extract runtime, error, and PSNR from saved results"""
    results = {}
    
    # Load original for error/PSNR calculation
    X_true_path = f'blurred_images/{img_name}_original.npy'
    if not os.path.exists(X_true_path):
        print(f"Warning: {X_true_path} not found")
        return None
    
    X_true = np.load(X_true_path)
    
    # Check each method
    methods = {
        'LU': ('lu', 'results'),
        'QR': ('qr', 'results'),
        'my_qr': ('qr_custom', 'results'),
        'LS': ('ls', 'results')
    }
    
    for method_name, (file_key, base_dir) in methods.items():
        result_path = f'{base_dir}/{img_name}_{config}_deblurred_{file_key}.npy'
        
        if os.path.exists(result_path):
            X_rec = np.load(result_path)
            error = compute_relative_error(X_rec, X_true)
            psnr = compute_psnr(X_rec, X_true)
            
            # Try to get runtime from a summary file or estimate
            # For now, we'll need to read from processing output or calculate
            results[method_name] = {
                'error': error,
                'psnr': psnr,
                'time': None  # Will need to be filled manually or from log
            }
            print(f"{img_name} {method_name}: error={error:.3e}, PSNR={psnr:.2f} dB")
        else:
            results[method_name] = None
            print(f"{img_name} {method_name}: Not found")
    
    return results

def main():
    test_cases = [
        ('1800_m8', 1800),
        ('2048_mountain', 2048),
    ]
    
    all_results = {}
    for img_name, img_size in test_cases:
        print(f"\n{'='*60}")
        print(f"Extracting results for {img_name} ({img_size}×{img_size})")
        print(f"{'='*60}")
        results = extract_results(img_name)
        if results:
            all_results[img_name] = results
    
    # Print summary for manual update
    print("\n" + "="*60)
    print("DATA FOR LINE CHART UPDATE")
    print("="*60)
    print("\nUpdate create_line_charts.py with the following data:\n")
    
    for img_name, img_size in test_cases:
        if img_name in all_results:
            size_key = f'{img_size}×{img_size}'
            print(f"# {size_key}:")
            for method in ['LU', 'QR', 'my_qr', 'LS']:
                if all_results[img_name].get(method):
                    data = all_results[img_name][method]
                    print(f"  '{size_key}': {{'{method}': {data.get('time', 'None')}, ...}},  # error={data['error']:.3e}, PSNR={data['psnr']:.2f}")

if __name__ == '__main__':
    main()

