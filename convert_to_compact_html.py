"""
Convert compact Markdown to HTML optimized for 12-page PDF
"""
import markdown
import re
from markdown.extensions import tables, fenced_code

md_file = "DDA3005_Final_Report_Compact.md"
html_file = "DDA3005_Final_Report_Compact.html"

with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Use markdown with HTML preservation
html_content = markdown.markdown(
    md_content, 
    extensions=['tables', 'fenced_code', 'codehilite'],
    extension_configs={
        'codehilite': {
            'use_pygments': False
        }
    }
)

# Post-process code blocks to add syntax highlighting for comments
def highlight_code_comments(html):
    """Add color highlighting to Python comments in code blocks"""
    def process_code_block(match):
        full_match = match.group(0)
        code_content = match.group(2)
        lang = match.group(1) if match.group(1) else ''
        
        # Only process Python code blocks
        if lang.strip() == 'python' or (not lang and ('def ' in code_content or 'import ' in code_content)):
            lines = code_content.split('\n')
            highlighted_lines = []
            
            for line in lines:
                # Check for comment (starts with #, possibly with leading whitespace)
                comment_match = re.match(r'^(\s*)(#.*)$', line)
                if comment_match:
                    indent = comment_match.group(1)
                    comment = comment_match.group(2)
                    # Escape HTML in comment text only
                    comment_escaped = comment.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    highlighted_lines.append(f'{indent}<span class="comment">{comment_escaped}</span>')
                else:
                    # For non-comment lines, just escape HTML but keep code as-is
                    # Escape HTML special characters
                    escaped_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    highlighted_lines.append(escaped_line)
            
            highlighted_code = '\n'.join(highlighted_lines)
            return f'<pre><code class="language-{lang}">{highlighted_code}</code></pre>'
        else:
            # Non-Python code, just escape HTML
            code_escaped = code_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            return f'<pre><code class="language-{lang}">{code_escaped}</code></pre>'
    
    # Match code blocks: <pre><code class="language-xxx">...</code></pre>
    # Pattern matches: <pre><code class="language-xxx">content</code></pre>
    pattern = r'<pre><code(?:\s+class="language-([^"]*)")?>([\s\S]*?)</code></pre>'
    html = re.sub(pattern, process_code_block, html)
    
    return html

# Apply syntax highlighting to code blocks
html_content = highlight_code_comments(html_content)

# Compact HTML with tight spacing for 12-page limit
html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DDA3005 Project Report</title>
    <style>
        @page {{
            size: A4;
            margin: 1.8cm 1.5cm;
        }}
        body {{
            font-family: 'Times New Roman', serif;
            line-height: 1.4;
            color: #1a1a1a;
            font-size: 10.5pt;
        }}
        h1 {{
            font-size: 18pt;
            margin-top: 0.8em;
            margin-bottom: 0.4em;
            page-break-after: avoid;
            color: #1a1a1a;
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 0.2em;
        }}
        h2 {{
            font-size: 14pt;
            margin-top: 0.6em;
            margin-bottom: 0.3em;
            page-break-after: avoid;
            color: #2c3e50;
            border-bottom: 1px solid #3498db;
            padding-bottom: 0.15em;
        }}
        h3 {{
            font-size: 12pt;
            margin-top: 0.5em;
            margin-bottom: 0.25em;
            page-break-after: avoid;
            color: #34495e;
        }}
        p {{
            margin: 0.4em 0;
            text-align: justify;
        }}
        strong, b {{
            color: #2c3e50;
            font-weight: bold;
        }}
        code {{
            font-family: 'Courier New', monospace;
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 2px;
            font-size: 9pt;
            border: 1px solid #e9ecef;
            color: #d63384;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 8px;
            border: 1px solid #dee2e6;
            border-left: 3px solid #3498db;
            border-radius: 3px;
            overflow-x: auto;
            page-break-inside: avoid;
            font-size: 9pt;
            margin: 0.5em 0;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            border: none;
            color: #2d2d2d;
            font-weight: normal;
        }}
        pre code .comment {{
            color: #008000;
            font-style: italic;
            font-weight: normal;
        }}
        pre code .string {{
            color: #ce9178;
        }}
        pre code .keyword {{
            color: #569cd6;
            font-weight: bold;
        }}
        pre code .function {{
            color: #dcdcaa;
        }}
        pre code .number {{
            color: #b5cea8;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 0.6em 0;
            page-break-inside: avoid;
            font-size: 9.5pt;
        }}
        th {{
            background: linear-gradient(to bottom, #3498db, #2980b9);
            color: white;
            font-weight: bold;
            padding: 6px;
            text-align: left;
            border: 1px solid #2980b9;
        }}
        td {{
            border: 1px solid #dee2e6;
            padding: 5px;
            text-align: left;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        img {{
            max-width: 100%;
            height: auto;
            page-break-inside: avoid;
            border: 1px solid #dee2e6;
            border-radius: 3px;
            margin: 0.5em 0;
        }}
        em {{
            font-style: italic;
            font-size: 9pt;
            color: #6c757d;
        }}
        ul, ol {{
            margin: 0.4em 0;
            padding-left: 1.5em;
        }}
        li {{
            margin: 0.2em 0;
        }}
        @media print {{
            body {{
                margin: 0;
            }}
            h1, h2, h3 {{
                page-break-after: avoid;
            }}
            img {{
                page-break-inside: avoid;
            }}
            table {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
{html_content}
</body>
</html>
"""

with open(html_file, 'w', encoding='utf-8') as f:
    f.write(html_doc)

print(f"Compact HTML file created: {html_file}")
print("Optimized for 12-page PDF with tight spacing and compact formatting.")

