"""
Create comparison charts for the report
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 10

# Data from experiments
images = ['512_car', '1024_books']
configs = ['config1', 'config2']

# Runtime data (seconds)
runtime_data = {
    '512_car': {
        'config1': {'LU': 0.080, 'QR': 0.025, 'LS': 1.354},
        'config2': {'LU': 0.060, 'QR': 0.027, 'LS': 1.396}
    },
    '1024_books': {
        'config1': {'LU': 2.016, 'QR': 0.137, 'LS': 7.353},
        'config2': {'LU': 1.963, 'QR': 0.158, 'LS': 6.883}
    }
}

# PSNR data (dB) - use finite values only
psnr_data = {
    '512_car': {
        'config1': {'LU': -1275.99, 'QR': -1095.50, 'LS': 76.27},
        'config2': {'LU': 200, 'QR': 200, 'LS': 200}  # Approximate for inf
    },
    '1024_books': {
        'config1': {'LU': -2748.43, 'QR': -2537.58, 'LS': 78.05},
        'config2': {'LU': 193.50, 'QR': 192.88, 'LS': 188.65}
    }
}

# Relative error data (log scale)
error_data = {
    '512_car': {
        'config1': {'LU': 66, 'QR': 57, 'LS': -1.1},  # log10 of error
        'config2': {'LU': -8, 'QR': -8, 'LS': -7.5}
    },
    '1024_books': {
        'config1': {'LU': 140, 'QR': 129, 'LS': -1.1},
        'config2': {'LU': -7, 'QR': -7, 'LS': -6.7}
    }
}

# Create runtime comparison chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
methods = ['LU', 'QR', 'LS']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, img in enumerate(images):
    ax = axes[idx]
    x = np.arange(len(methods))
    width = 0.35
    
    config1_times = [runtime_data[img]['config1'][m] for m in methods]
    config2_times = [runtime_data[img]['config2'][m] for m in methods]
    
    bars1 = ax.bar(x - width/2, config1_times, width, label='Config1', color=colors[0], alpha=0.7)
    bars2 = ax.bar(x + width/2, config2_times, width, label='Config2', color=colors[1], alpha=0.7)
    
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontweight='bold')
    ax.set_title(f'Runtime Comparison - {img}', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}s',
                   ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/runtime_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create PSNR comparison chart (only for config2 and config1 LS)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, img in enumerate(images):
    ax = axes[idx]
    
    # Config2 PSNR (all methods)
    config2_psnr = [psnr_data[img]['config2'][m] for m in methods]
    
    # Config1 PSNR (only LS, others are negative)
    config1_psnr = [psnr_data[img]['config1']['LS'] if m == 'LS' else None for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [p if p else 0 for p in config1_psnr], width, 
                   label='Config1 (LS only)', color=colors[2], alpha=0.7)
    bars2 = ax.bar(x + width/2, config2_psnr, width, 
                   label='Config2', color=colors[1], alpha=0.7)
    
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('PSNR (dB)', fontweight='bold')
    ax.set_title(f'PSNR Comparison - {img}', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 220])
    
    # Add value labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if config1_psnr[i] is not None:
            ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
                   f'{config1_psnr[i]:.1f}',
                   ha='center', va='bottom', fontsize=8)
        ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height(),
               f'{config2_psnr[i]:.1f}',
               ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('results/psnr_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create relative error comparison (log scale)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, img in enumerate(images):
    ax = axes[idx]
    
    config1_errors = [error_data[img]['config1'][m] for m in methods]
    config2_errors = [error_data[img]['config2'][m] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, config1_errors, width, 
                   label='Config1', color=colors[0], alpha=0.7)
    bars2 = ax.bar(x + width/2, config2_errors, width, 
                   label='Config2', color=colors[1], alpha=0.7)
    
    ax.set_xlabel('Method', fontweight='bold')
    ax.set_ylabel('log₁₀(Relative Error)', fontweight='bold')
    ax.set_title(f'Relative Error Comparison - {img}', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            label = f'{height:.1f}' if height > 0 else f'{height:.1f}'
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   label,
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=8)

plt.tight_layout()
plt.savefig('results/error_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create speedup comparison (QR vs LU)
fig, ax = plt.subplots(figsize=(10, 6))

speedup_data = {
    '512_car': {'config1': 0.080/0.025, 'config2': 0.060/0.027},
    '1024_books': {'config1': 2.016/0.137, 'config2': 1.963/0.158}
}

x = np.arange(len(images))
width = 0.35

config1_speedup = [speedup_data[img]['config1'] for img in images]
config2_speedup = [speedup_data[img]['config2'] for img in images]

bars1 = ax.bar(x - width/2, config1_speedup, width, label='Config1', color=colors[0], alpha=0.7)
bars2 = ax.bar(x + width/2, config2_speedup, width, label='Config2', color=colors[1], alpha=0.7)

ax.set_xlabel('Image', fontweight='bold')
ax.set_ylabel('Speedup (QR vs LU)', fontweight='bold')
ax.set_title('QR Speedup over LU', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(images)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}×',
               ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('results/speedup_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Created comparison charts:")
print("  - results/runtime_comparison.png")
print("  - results/psnr_comparison.png")
print("  - results/error_comparison.png")
print("  - results/speedup_comparison.png")

