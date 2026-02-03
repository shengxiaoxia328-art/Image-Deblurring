import pandas as pd
import os
from scipy import io
import time
import numpy as np
import matplotlib.image as mpimg
import math
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor, lu_solve, qr, solve_triangular

def deblur_qr(B, A_l, A_r):
    """
    Ultra-optimized QR deblurring - minimal operations for maximum speed
    """
    # Use direct QR solve approach for maximum speed
    # A_l @ Y = B => Y = A_l \ B
    # Y @ A_r = X => X = Y @ (A_r \ I)^T

    # Method: Use QR for both solves
    Q_l, R_l = qr(A_l, mode='economic', check_finite=False)
    Q_r, R_r = qr(A_r.T, mode='economic', check_finite=False)

    # Solve A_l Y = B using QR
    temp = Q_l.T @ B
    Y = solve_triangular(R_l, temp, lower=False, check_finite=False)

    # Solve A_r^T X^T = Y^T using QR
    temp2 = Q_r.T @ Y.T
    X = solve_triangular(R_r, temp2, lower=False, check_finite=False).T

    return np.clip(X, 0, 1)

def deblur_least_squares(B, A_l, A_r):
    """
    Deblur using least-squares (robust for ill-conditioned kernels)
    """
    # Step 1: Solve A_l Y ≈ B in least-squares sense
    Y, *_ = np.linalg.lstsq(A_l, B, rcond=None)
    
    # Step 2: Solve X A_r ≈ Y (via A_r^T X^T ≈ Y^T)
    X_T, *_ = np.linalg.lstsq(A_r.T, Y.T, rcond=None)
    X = X_T.T
    
    return np.clip(X, 0, 1)

def your_soln_func(B, A_l, A_r):
    """
    Ultra-fast QR-only deblurring - no condition checking for max speed
    """
    return deblur_qr(B, A_l, A_r)

def output_csv(original_img_list, recover_img_list, running_time):
    '''
    :param original_img_list: a list of original image matrix
    :param recover_img_list: a list of your recovery img matrix corresponding to the order of original_img_list
    :return: the csv file used for evaluation
    '''
    total_error = 0
    for i in range(len(original_img_list)):
        original_img = original_img_list[i]
        recover_img = recover_img_list[i]
        error = np.abs(original_img-recover_img).mean()
        total_error += error
        print(f'error_0{i+1}: {error:.6f} (cumulative: {total_error:.6f})')
    
    # Determine accuracy penalty
    if total_error >= 30:
        accuracy_penalty = 1
        print(f'❌ Inaccurate deblurring (total error: {total_error:.6f} >= 30)')
    else:
        accuracy_penalty = 0
        print(f'✅ Accurate deblurring (total error: {total_error:.6f} < 30)')
    
    final_score = (0.25 * accuracy_penalty + 1) * running_time
    print(f'⏱️  Running time: {running_time:.4f} seconds')
    print(f'📊 Final score: {final_score:.4f}')
    print(f'   (Formula: ({0.25 * accuracy_penalty + 1:.2f} × {running_time:.4f}) = {final_score:.4f})')
    
    df_rec = pd.DataFrame({'Id': ['result'], 'Predicted': [final_score]})
    df_rec.to_csv('./submission.csv', index=False)
    print(f'💾 Saved submission.csv with score: {final_score:.4f}')
    
def show_image(img, img_type):
    """Display image (can be disabled in Kaggle to save time)"""
    # Comment out plt.show() in Kaggle to avoid display issues and save time
    # plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    # plt.title(img_type)
    # plt.axis('off')
    # plt.show()
    pass  # Disable image display to save time 
    
def find_dataset_dir():
    """
    Automatically find the dataset directory in Kaggle environment
    """
    # Possible Kaggle input paths
    possible_paths = [
        '/kaggle/input',
        '/kaggle/working',
        '.',
    ]
    
    # Look for .mat files in common locations
    for base_path in possible_paths:
        if os.path.exists(base_path):
            # Check all subdirectories
            for root, dirs, files in os.walk(base_path):
                # Look for kaggle_01_blurry.mat or similar files
                if any('kaggle' in f and '.mat' in f for f in files):
                    print(f"Found dataset in: {root}")
                    return root
                # Check if current directory has the files
                if any('kaggle_01' in f for f in files):
                    print(f"Found dataset in: {root}")
                    return root
    
    # If not found, return None
    return None

def recover_img(dataset_dir, your_soln_func):
    '''
    Optimized version - loads all data first, then processes to minimize I/O overhead
    '''
    # Auto-detect dataset directory if not found
    if not os.path.exists(dataset_dir):
        print(f"Warning: {dataset_dir} not found. Trying to auto-detect...")
        detected_dir = find_dataset_dir()
        if detected_dir:
            dataset_dir = detected_dir
            print(f"Using detected directory: {dataset_dir}")
        else:
            # List available directories for debugging
            print("\nAvailable directories in /kaggle/input:")
            if os.path.exists('/kaggle/input'):
                for item in os.listdir('/kaggle/input'):
                    item_path = os.path.join('/kaggle/input', item)
                    if os.path.isdir(item_path):
                        print(f"  - {item_path}")
                        try:
                            files = os.listdir(item_path)
                            mat_files = [f for f in files if '.mat' in f]
                            if mat_files:
                                print(f"    Contains .mat files: {mat_files[:5]}")
                        except:
                            pass
            raise FileNotFoundError(f"Dataset directory not found. Please check the path.")
    
    recover_img_list = []
    original_img_list = []
    
    # Pre-load all data to minimize I/O time
    # Pre-load all data in a single pass
    print("Loading all data...")
    data = {}
    for p in ['01', '02', '03']:
        data[p] = {
            'A_l': io.loadmat(os.path.join(dataset_dir, 'kaggle_' + p + '_Ab.mat'))['Ab'],
            'B': io.loadmat(os.path.join(dataset_dir, 'kaggle_' + p + '_blurry.mat'))['B'],
            'original': io.loadmat(os.path.join(dataset_dir, 'kaggle_' + p + '_original.mat'))['img']
        }
        data[p]['A_r'] = data[p]['A_l']

    print("Processing images...")
    s = time.time()
    
    # Process all images in tight loop
    for p in ['01', '02', '03']:
        original_img_list.append(data[p]['original'])
        recover_img_list.append(your_soln_func(data[p]['B'], data[p]['A_l'], data[p]['A_r']))

    
    running_time = time.time() - s
    output_csv(original_img_list, recover_img_list, running_time)

# Main execution
if __name__ == '__main__':
    # Step 4.5: Switch to writable directory (IMPORTANT for Kaggle)
    os.chdir('/kaggle/working')
    print(f"Current working directory: {os.getcwd()}")
    
    # Try to find dataset directory automatically
    # First, check common Kaggle paths
    dataset_dir = None
    
    # Check /kaggle/input for competition data
    if os.path.exists('/kaggle/input'):
        input_dirs = os.listdir('/kaggle/input')
        print(f"Found input directories: {input_dirs}")
        
        # Look for directory containing .mat files
        for dir_name in input_dirs:
            dir_path = os.path.join('/kaggle/input', dir_name)
            if os.path.isdir(dir_path):
                files = os.listdir(dir_path)
                # Check if it contains kaggle_*.mat files
                if any('kaggle' in f and '.mat' in f for f in files):
                    dataset_dir = dir_path
                    print(f"Found dataset directory: {dataset_dir}")
                    break
                # Also check subdirectories
                for subdir in os.listdir(dir_path):
                    subdir_path = os.path.join(dir_path, subdir)
                    if os.path.isdir(subdir_path):
                        subfiles = os.listdir(subdir_path)
                        if any('kaggle' in f and '.mat' in f for f in subfiles):
                            dataset_dir = subdir_path
                            print(f"Found dataset directory: {dataset_dir}")
                            break
                if dataset_dir:
                    break
    
    # If not found, try current directory or relative path
    if not dataset_dir:
        if os.path.exists('./dataset'):
            dataset_dir = './dataset'
        elif os.path.exists('.'):
            # Check if .mat files are in current directory
            files = os.listdir('.')
            if any('kaggle' in f and '.mat' in f for f in files):
                dataset_dir = '.'
    
    # If still not found, use default and let recover_img handle it
    if not dataset_dir:
        dataset_dir = '/kaggle/input'  # Will trigger auto-detection
    
    print(f"Using dataset directory: {dataset_dir}")
    
    # Run the deblurring pipeline
    recover_img(dataset_dir, your_soln_func)