"""
Convert Markdown to HTML with enhanced styling for PDF
"""
import markdown
from markdown.extensions import tables, fenced_code

md_file = "DDA3005_Final_Report.md"
html_file = "DDA3005_Final_Report.html"

with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML
html_content = markdown.markdown(
    md_content, 
    extensions=['tables', 'fenced_code', 'codehilite']
)

# Enhanced HTML document with professional styling
html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DDA3005 Project Report - Image Deblurring and QR Factorizations</title>
    <style>
        @page {{
            size: A4;
            margin: 2.5cm 2cm;
        }}
        body {{
            font-family: 'Times New Roman', 'SimSun', serif;
            line-height: 1.7;
            color: #1a1a1a;
            max-width: 100%;
            background: white;
        }}
        h1 {{
            font-size: 24pt;
            margin-top: 1.2em;
            margin-bottom: 0.6em;
            page-break-after: avoid;
            color: #1a1a1a;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 0.3em;
        }}
        h2 {{
            font-size: 20pt;
            margin-top: 1em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.2em;
        }}
        h3 {{
            font-size: 16pt;
            margin-top: 0.8em;
            margin-bottom: 0.4em;
            page-break-after: avoid;
            color: #34495e;
        }}
        p {{
            margin: 0.6em 0;
            text-align: justify;
            text-indent: 0;
        }}
        strong, b {{
            color: #2c3e50;
            font-weight: bold;
        }}
        code {{
            font-family: 'Courier New', 'Consolas', monospace;
            background-color: #f8f9fa;
            padding: 3px 6px;
            border-radius: 3px;
            font-size: 0.9em;
            border: 1px solid #e9ecef;
            color: #d63384;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 15px;
            border: 1px solid #dee2e6;
            border-left: 4px solid #3498db;
            border-radius: 5px;
            overflow-x: auto;
            page-break-inside: avoid;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
            border: none;
            color: #212529;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1.2em 0;
            page-break-inside: avoid;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(to bottom, #3498db, #2980b9);
            color: white;
            font-weight: bold;
            padding: 12px;
            text-align: left;
            border: 1px solid #2980b9;
        }}
        td {{
            border: 1px solid #dee2e6;
            padding: 10px;
            text-align: left;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e9ecef;
        }}
        img {{
            max-width: 100%;
            height: auto;
            page-break-inside: avoid;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 1em 0;
        }}
        .figure {{
            text-align: center;
            margin: 1.5em 0;
            page-break-inside: avoid;
        }}
        .figure-caption {{
            font-style: italic;
            margin-top: 0.5em;
            font-size: 0.9em;
            color: #6c757d;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            padding-left: 1.5em;
            margin: 1em 0;
            color: #495057;
            background-color: #f8f9fa;
            padding: 1em 1.5em;
            border-radius: 0 5px 5px 0;
        }}
        ul, ol {{
            margin: 0.8em 0;
            padding-left: 2em;
        }}
        li {{
            margin: 0.4em 0;
        }}
        /* Highlight key formulas */
        p:has(strong:contains("=")) {{
            background-color: #fff9e6;
            padding: 0.5em;
            border-left: 3px solid #ffc107;
            margin: 1em 0;
        }}
        /* Matrix and formula styling */
        p strong {{
            font-size: 1.05em;
            letter-spacing: 0.5px;
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

print(f"✓ Enhanced HTML file created: {html_file}")
print("\nTo convert to PDF:")
print("1. Open the HTML file in your browser (Chrome/Edge recommended)")
print("2. Press Ctrl+P (or File > Print)")
print("3. Select 'Save as PDF' as the destination")
print("4. Set margins to 'Minimum' or 'None'")
print("5. Enable 'Background graphics' for best appearance")
print("6. Click 'Save'")

