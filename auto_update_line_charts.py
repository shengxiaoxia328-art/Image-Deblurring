"""
Automatically update line chart data from processing summary
"""
import re

def parse_summary_file(summary_file='results/large_image_summary.txt'):
    """Parse summary file and extract data"""
    data = {}
    
    try:
        with open(summary_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header lines
        for line in lines[3:]:  # Skip title, separator, header
            if '---' in line or not line.strip():
                continue
            
            # Parse: img_name method time error psnr
            parts = line.split()
            if len(parts) >= 5:
                img_name = parts[0]
                method = parts[1]
                time = float(parts[2])
                error = float(parts[3])
                psnr = float(parts[4])
                
                # Extract size from img_name (e.g., 1800_m8 -> 1800)
                size_match = re.match(r'(\d+)_', img_name)
                if size_match:
                    size = int(size_match.group(1))
                    size_key = f'{size}×{size}'
                    
                    if size_key not in data:
                        data[size_key] = {}
                    data[size_key][method] = {
                        'time': time,
                        'error': error,
                        'psnr': psnr
                    }
        
        return data
    except FileNotFoundError:
        print(f"Summary file {summary_file} not found. Processing may not be complete.")
        return None
    except Exception as e:
        print(f"Error parsing summary file: {e}")
        return None

def update_line_chart_script(data):
    """Update create_line_charts.py with new data"""
    if data is None:
        print("No data to update.")
        return
    
    # Read current script
    with open('create_line_charts.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update runtime_config2
    for size_key, methods_data in data.items():
        for method, values in methods_data.items():
            # Find and replace runtime
            pattern = f"'{size_key}': {{'LU': None, 'QR': None, 'my_qr': None, 'LS': None}}"
            if pattern in content:
                # Replace with actual values
                new_dict = f"'{size_key}': {{"
                for m in ['LU', 'QR', 'my_qr', 'LS']:
                    if m in methods_data:
                        new_dict += f"'{m}': {methods_data[m]['time']:.4f}, "
                    else:
                        new_dict += f"'{m}': None, "
                new_dict = new_dict.rstrip(', ') + "}"
                content = content.replace(pattern, new_dict)
            
            # Update PSNR
            pattern_psnr = f"'{size_key}': {{'LU': None, 'QR': None, 'my_qr': None, 'LS': None}}"
            if pattern_psnr in content and 'psnr_config2' in content:
                # Find psnr_config2 section
                psnr_start = content.find("psnr_config2 = {")
                if psnr_start != -1:
                    # Similar replacement for PSNR
                    pass  # Will implement if needed
    
    # For now, print the data that needs to be manually updated
    print("\n" + "="*60)
    print("DATA TO UPDATE IN create_line_charts.py")
    print("="*60)
    
    for size_key, methods_data in data.items():
        print(f"\n# {size_key}:")
        print(f"runtime_config2['{size_key}'] = {{")
        for m in ['LU', 'QR', 'my_qr', 'LS']:
            if m in methods_data:
                print(f"    '{m}': {methods_data[m]['time']:.4f},  # error={methods_data[m]['error']:.3e}, PSNR={methods_data[m]['psnr']:.2f} dB")
            else:
                print(f"    '{m}': None,")
        print("}")
        
        print(f"\npsnr_config2['{size_key}'] = {{")
        for m in ['LU', 'QR', 'my_qr', 'LS']:
            if m in methods_data:
                print(f"    '{m}': {methods_data[m]['psnr']:.2f},")
            else:
                print(f"    '{m}': None,")
        print("}")
        
        print(f"\nerror_config2['{size_key}'] = {{")
        for m in ['LU', 'QR', 'my_qr', 'LS']:
            if m in methods_data:
                error_log10 = -np.log10(methods_data[m]['error']) if methods_data[m]['error'] > 0 else None
                print(f"    '{m}': {error_log10:.2f},")
            else:
                print(f"    '{m}': None,")
        print("}")

def main():
    import numpy as np
    data = parse_summary_file()
    if data:
        update_line_chart_script(data)
        print("\n" + "="*60)
        print("Please manually update create_line_charts.py with the data above,")
        print("then run: python create_line_charts.py")
        print("="*60)
    else:
        print("Waiting for processing to complete...")
        print("Run: python process_large_images.py")

if __name__ == '__main__':
    import numpy as np
    main()

