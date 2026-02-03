import os

files = ['runtime_comparison.png', 'psnr_comparison.png', 'error_comparison.png', 'speedup_comparison.png']
print('Checking image files:')
for f in files:
    path = f'results/{f}'
    exists = os.path.exists(path)
    print(f'  {f}: {exists}')

