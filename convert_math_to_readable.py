"""
Convert LaTeX math formulas to readable text format
"""
import re

def convert_latex_to_readable(text):
    """Convert LaTeX math to readable format"""
    
    # Replace display math \[...\]
    text = re.sub(r'\\\[(.*?)\\\]', lambda m: convert_math_expr(m.group(1)), text, flags=re.DOTALL)
    
    # Replace inline math \(...\)
    text = re.sub(r'\\\((.*?)\\\)', lambda m: convert_math_expr(m.group(1)), text, flags=re.DOTALL)
    
    # Replace $$...$$
    text = re.sub(r'\$\$(.*?)\$\$', lambda m: convert_math_expr(m.group(1)), text, flags=re.DOTALL)
    
    # Replace $...$
    text = re.sub(r'\$(.*?)\$', lambda m: convert_math_expr(m.group(1)), text, flags=re.DOTALL)
    
    return text

def convert_math_expr(expr):
    """Convert a single math expression"""
    expr = expr.strip()
    
    # Replace common symbols
    expr = expr.replace('\\ell', 'ℓ')
    expr = expr.replace('\\mathbb{R}', 'R')
    expr = expr.replace('\\mathbb{R}^{', 'R^')
    expr = expr.replace('\\times', '×')
    expr = expr.replace('\\ldots', '...')
    expr = expr.replace('\\quad', ' ')
    expr = expr.replace('\\min', 'min')
    expr = expr.replace('\\max', 'max')
    expr = expr.replace('\\|', '||')
    expr = expr.replace('_F', '_F')
    
    # Handle superscripts: ^{n} -> ^n, ^{n \times n} -> ^(n×n)
    def replace_superscript(m):
        content = m.group(1)
        content = content.replace('\\times', '×')
        content = content.replace(' ', '')
        if '×' in content or len(content) > 1:
            return f'^({content})'
        return f'^{content}'
    
    expr = re.sub(r'\^{([^}]+)}', replace_superscript, expr)
    
    # Handle subscripts: _{n} -> _n, _{n-1} -> _{n-1}
    def replace_subscript(m):
        content = m.group(1)
        return f'_{content}'
    
    expr = re.sub(r'_{([^}]+)}', replace_subscript, expr)
    
    # Handle fractions: \frac{a}{b} -> (a/b)
    expr = re.sub(r'\\frac{([^}]+)}{([^}]+)}', r'(\1/\2)', expr)
    
    # Handle inverse: ^{-1} -> ^(-1)
    expr = expr.replace('^{-1}', '^(-1)')
    
    # Handle transpose: ^T -> ^T (keep as is)
    # Already handled by superscript regex
    
    # Clean up extra spaces
    expr = re.sub(r'\s+', ' ', expr)
    expr = expr.strip()
    
    return expr

# Read the markdown file
with open('DDA3005_Final_Report.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Convert all math expressions
content = convert_latex_to_readable(content)

# Write back
with open('DDA3005_Final_Report.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Converted all LaTeX math formulas to readable format")

