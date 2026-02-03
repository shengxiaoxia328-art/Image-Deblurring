"""
Create comparison figures showing all five methods (LU, QR, my_qr, LS, Padding) together:
1. Deblurred images comparison
2. Error maps comparison
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'Times New Roman'

def load_deblurred_image(img_name, config, method):
    """Load deblurred image from saved .npy file"""
    if method == 'my_qr':
        method = 'qr_custom'
    elif method == 'QR':
        method = 'qr'
    elif method == 'LS':
        method = 'ls'
    elif method == 'LU':
        method = 'lu'
    elif method == 'Padding' or method == 'PAD':
        method = 'pad'
    
    try:
        return np.load(f'results/{img_name}_{config}_deblurred_{method}.npy')
    except FileNotFoundError:
        print(f"Warning: {img_name}_{config}_deblurred_{method}.npy not found")
        return None

def compute_error_map(X_rec, X_true):
    """Compute normalized error map for a single method.

    We normalize each error map by its own maximum and apply a mild
    square-root to enhance low-amplitude structures, so shapes remain
    visible even when the overall error is very small.
    """
    diff = np.abs(X_rec - X_true)
    max_val = diff.max()
    if max_val > 0:
        diff = diff / max_val
        diff = np.sqrt(diff)
    return diff

def create_deblurred_comparison(img_name, config):
    """Create figure with all five deblurred images"""
    X_true = np.load(f'blurred_images/{img_name}_original.npy')
    B = np.load(f'blurred_images/{img_name}_blurred_{config}.npy')
    
    # Load all five methods
    X_lu = load_deblurred_image(img_name, config, 'LU')
    X_qr = load_deblurred_image(img_name, config, 'QR')
    X_myqr = load_deblurred_image(img_name, config, 'my_qr')
    X_ls = load_deblurred_image(img_name, config, 'LS')
    X_pad = load_deblurred_image(img_name, config, 'Padding')
    
    if X_lu is None or X_qr is None or X_myqr is None or X_ls is None or X_pad is None:
        print(f"Skipping {img_name} {config} - missing some results")
        return
    
    # Create figure: 2 rows x 3 columns
    # Row 1: Original, Blurred, LU
    # Row 2: QR, my_qr, LS, Padding
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Row 1
    axes[0, 0].imshow(X_true, cmap='gray')
    axes[0, 0].set_title('Original', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(B, cmap='gray')
    axes[0, 1].set_title('Blurred', fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(X_lu, cmap='gray')
    axes[0, 2].set_title('LU', fontweight='bold', fontsize=11)
    axes[0, 2].axis('off')
    
    # Row 2
    axes[1, 0].imshow(X_qr, cmap='gray')
    axes[1, 0].set_title('QR (SciPy)', fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(X_myqr, cmap='gray')
    axes[1, 1].set_title('QR (my_qr)', fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(X_ls, cmap='gray')
    axes[1, 2].set_title('LS', fontweight='bold', fontsize=11)
    axes[1, 2].axis('off')
    
    # Add Padding as a separate figure or combine
    # For now, we'll create a 3x2 layout: Row 1: Original, Blurred, LU; Row 2: QR, my_qr, LS; Row 3: Padding (centered)
    # Actually, let's use 2x3 and add padding below or create a new layout
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f'results/{img_name}_{config}_five_methods_deblurred.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Create a separate figure showing all 5 methods + padding
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 9))
    axes2[0, 0].imshow(X_true, cmap='gray')
    axes2[0, 0].set_title('Original', fontweight='bold', fontsize=11)
    axes2[0, 0].axis('off')
    
    axes2[0, 1].imshow(B, cmap='gray')
    axes2[0, 1].set_title('Blurred', fontweight='bold', fontsize=11)
    axes2[0, 1].axis('off')
    
    axes2[0, 2].imshow(X_lu, cmap='gray')
    axes2[0, 2].set_title('LU', fontweight='bold', fontsize=11)
    axes2[0, 2].axis('off')
    
    axes2[1, 0].imshow(X_qr, cmap='gray')
    axes2[1, 0].set_title('QR (SciPy)', fontweight='bold', fontsize=11)
    axes2[1, 0].axis('off')
    
    axes2[1, 1].imshow(X_ls, cmap='gray')
    axes2[1, 1].set_title('LS', fontweight='bold', fontsize=11)
    axes2[1, 1].axis('off')
    
    axes2[1, 2].imshow(X_pad, cmap='gray')
    axes2[1, 2].set_title('Padding', fontweight='bold', fontsize=11)
    axes2[1, 2].axis('off')
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f'results/{img_name}_{config}_five_methods_deblurred.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Created {img_name}_{config}_five_methods_deblurred.png")

def create_error_map_comparison(img_name, config):
    """Create figure with error maps for all five methods"""
    X_true = np.load(f'blurred_images/{img_name}_original.npy')
    
    # Load all five methods
    X_lu = load_deblurred_image(img_name, config, 'LU')
    X_qr = load_deblurred_image(img_name, config, 'QR')
    X_myqr = load_deblurred_image(img_name, config, 'my_qr')
    X_ls = load_deblurred_image(img_name, config, 'LS')
    X_pad = load_deblurred_image(img_name, config, 'Padding')
    
    if X_lu is None or X_qr is None or X_myqr is None or X_ls is None or X_pad is None:
        print(f"Skipping {img_name} {config} - missing some results")
        return
    
    # Compute error maps (raw differences)
    err_norm = {
        'LU': compute_error_map(X_lu, X_true),
        'QR (SciPy)': compute_error_map(X_qr, X_true),
        'QR (my_qr)': compute_error_map(X_myqr, X_true),
        'LS': compute_error_map(X_ls, X_true),
        'Padding': compute_error_map(X_pad, X_true),
    }
    
    # Create figure: 2 rows x 3 columns (five error maps + one empty)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Row 1
    im1 = axes[0, 0].imshow(err_norm['LU'], cmap='hot', vmin=0, vmax=1)
    axes[0, 0].set_title('LU Error Map', fontweight='bold', fontsize=11)
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 1].imshow(err_norm['QR (SciPy)'], cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('QR (SciPy) Error Map', fontweight='bold', fontsize=11)
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im3 = axes[0, 2].imshow(err_norm['QR (my_qr)'], cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('QR (my_qr) Error Map', fontweight='bold', fontsize=11)
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2
    im4 = axes[1, 0].imshow(err_norm['LS'], cmap='hot', vmin=0, vmax=1)
    axes[1, 0].set_title('LS Error Map', fontweight='bold', fontsize=11)
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im5 = axes[1, 1].imshow(err_norm['Padding'], cmap='hot', vmin=0, vmax=1)
    axes[1, 1].set_title('Padding Error Map', fontweight='bold', fontsize=11)
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    axes[1, 2].axis('off')  # Empty space
    
    plt.tight_layout(pad=0.5)
    plt.savefig(f'results/{img_name}_{config}_five_methods_error_maps.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Created {img_name}_{config}_five_methods_error_maps.png")

def main():
    test_cases = [
        ('512_car', 'config1'),
        ('512_car', 'config2'),
        ('1024_books', 'config1'),
        ('1024_books', 'config2'),
    ]
    
    for img_name, config in test_cases:
        print(f"\nProcessing {img_name} {config}...")
        create_deblurred_comparison(img_name, config)
        create_error_map_comparison(img_name, config)
    
    print("\n✓ All comparison figures created!")

if __name__ == '__main__':
    main()

