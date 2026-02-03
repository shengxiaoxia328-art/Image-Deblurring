"""
Task a: Create blurring kernels A_ℓ and A_r and generate blurred images B
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def create_blurring_kernel(n, j, k):
    """
    Create a blurring kernel A according to the specification:
    [a_{n+j}, a_{n+j-1}, ..., a_{n+j-k+1}] = 2/(k(k+1)) * [k, k-1, ..., 1]
    and a_i = 0 for all i > n+j and i < n+j-k+1
    
    The matrix A has a Toeplitz structure where:
    - Row i starts with a_{n+i-1} and continues with a_{n+i-2}, ..., a_i
    - This creates a matrix where each row is shifted
    
    Parameters:
    n: size of the matrix (n×n)
    j: parameter j (shift parameter)
    k: parameter k (number of coefficients)
    
    Returns:
    A: n×n blurring kernel matrix
    """
    A = np.zeros((n, n))
    
    # Calculate the coefficients: [k, k-1, ..., 1] normalized by 2/(k(k+1))
    coeffs = np.array([k - i for i in range(k)])  # [k, k-1, ..., 1]
    coeffs = (2.0 / (k * (k + 1))) * coeffs
    
    # Create coefficient array a of length 2n-1
    # Index mapping: a[i] corresponds to a_{i+1} in the notation
    a = np.zeros(2 * n - 1)
    
    # Fill the coefficients according to the specification
    start_idx = n + j - k + 1  # Starting index: n+j-k+1
    for idx, coeff in enumerate(coeffs):
        pos = start_idx + idx - 1  # Convert to 0-based indexing
        if 0 <= pos < len(a):
            a[pos] = coeff
    
    # Build the Toeplitz matrix
    # Row i uses coefficients a[n+i-1], a[n+i-2], ..., a[i]
    for i in range(n):
        for col in range(n):
            # The coefficient index in array a
            a_idx = n + i - 1 - col
            if 0 <= a_idx < len(a):
                A[i, col] = a[a_idx]
    
    return A

def create_symmetric_banded_kernel(n, num_bands=10):
    """
    Create a symmetric banded blurring kernel as specified:
    a_n = 10/100, a_{n-1} = 9/100, ..., a_{n-9} = 1/100
    a_i = 0 for i < n-9
    
    Parameters:
    n: size of the matrix
    num_bands: number of bands (default 10)
    
    Returns:
    A: n×n symmetric banded blurring kernel
    """
    A = np.zeros((n, n))
    
    # Fill the main diagonal and upper diagonals
    for i in range(num_bands):
        coeff = (10 - i) / 100.0
        # Main diagonal (when i=0) and upper diagonals
        for row in range(n):
            col = row + i
            if col < n:
                A[row, col] = coeff
                # Make symmetric
                A[col, row] = coeff
    
    return A

def blur_image(X, A_l, A_r):
    """
    Apply blurring operation: B = A_ℓ X A_r
    
    Parameters:
    X: original image (n×n)
    A_l: left blurring kernel (n×n)
    A_r: right blurring kernel (n×n)
    
    Returns:
    B: blurred image (n×n)
    """
    B = A_l @ X @ A_r
    return B

def load_image(image_path):
    """
    Load and preprocess an image
    
    Parameters:
    image_path: path to the image file
    
    Returns:
    img: normalized image array (0-1 range)
    """
    im = Image.open(image_path)
    img = np.array(im)
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    
    # Normalize to [0, 1]
    img = img.astype(np.float64) / 255.0
    
    return img

def main():
    # Create output directory
    os.makedirs('blurred_images', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Select test images (at least two different images)
    test_images = [
        'test_images/512_car_original.png',
        'test_images/1024_books_original.png'
    ]
    
    for img_path in test_images:
        print(f"\nProcessing {img_path}...")
        
        # Load original image
        X = load_image(img_path)
        n = X.shape[0]
        print(f"Image size: {n}×{n}")
        
        # Create two different blurring configurations
        
        # Configuration 1: As specified in the PDF
        # A_ℓ: j=0, k=12 (upper triangular)
        # A_r: j=1, k=36 (upper triangular with extra subdiagonal)
        print("\nCreating Configuration 1...")
        A_l1 = create_blurring_kernel(n, j=0, k=min(12, n))  # Limit k to n
        A_r1 = create_blurring_kernel(n, j=1, k=min(36, n))  # Limit k to n
        
        # For stability, ensure matrices are well-conditioned
        # Add regularization to make matrices invertible
        # Use a stronger regularization for better numerical stability
        reg_strength = max(1e-6, 1e-8 * n)  # Scale with image size
        A_l1 = A_l1 + reg_strength * np.eye(n)
        A_r1 = A_r1 + reg_strength * np.eye(n)
        
        # Normalize rows to ensure stability
        row_sums_l = np.sum(A_l1, axis=1, keepdims=True)
        row_sums_l[row_sums_l == 0] = 1
        A_l1 = A_l1 / row_sums_l
        
        row_sums_r = np.sum(A_r1, axis=1, keepdims=True)
        row_sums_r[row_sums_r == 0] = 1
        A_r1 = A_r1 / row_sums_r
        
        B1 = blur_image(X, A_l1, A_r1)
        
        # Save blurred image
        img_name = os.path.basename(img_path).replace('_original.png', '')
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(X, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(B1, cmap='gray')
        plt.title('Blurred Image (Config 1)')
        plt.axis('off')
        
        # Configuration 2: Symmetric banded kernels
        print("Creating Configuration 2...")
        A_l2 = create_symmetric_banded_kernel(n, num_bands=10)
        A_r2 = create_symmetric_banded_kernel(n, num_bands=10)
        
        # Normalize
        A_l2 = A_l2 / (np.sum(A_l2) + 1e-10)
        A_r2 = A_r2 / (np.sum(A_r2) + 1e-10)
        
        B2 = blur_image(X, A_l2, A_r2)
        
        plt.subplot(1, 3, 3)
        plt.imshow(B2, cmap='gray')
        plt.title('Blurred Image (Config 2)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'results/{img_name}_blurred_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save blurred images and kernels
        np.save(f'blurred_images/{img_name}_original.npy', X)
        np.save(f'blurred_images/{img_name}_blurred_config1.npy', B1)
        np.save(f'blurred_images/{img_name}_blurred_config2.npy', B2)
        np.save(f'blurred_images/{img_name}_A_l1.npy', A_l1)
        np.save(f'blurred_images/{img_name}_A_r1.npy', A_r1)
        np.save(f'blurred_images/{img_name}_A_l2.npy', A_l2)
        np.save(f'blurred_images/{img_name}_A_r2.npy', A_r2)
        
        # Save as images
        plt.imsave(f'blurred_images/{img_name}_original.png', X, cmap='gray')
        plt.imsave(f'blurred_images/{img_name}_blurred_config1.png', B1, cmap='gray')
        plt.imsave(f'blurred_images/{img_name}_blurred_config2.png', B2, cmap='gray')
        
        print(f"Saved results for {img_name}")
    
    print("\nTask a completed! Blurred images and kernels saved.")

if __name__ == '__main__':
    main()

