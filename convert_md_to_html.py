"""
Convert Markdown to HTML (can be printed to PDF from browser)
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

# Create full HTML document with styling
html_doc = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DDA3005 Project Report - Image Deblurring and QR Factorizations</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
        }}
        body {{
            font-family: 'Times New Roman', 'SimSun', serif;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
        }}
        h1 {{
            font-size: 24pt;
            margin-top: 1em;
            margin-bottom: 0.5em;
            page-break-after: avoid;
        }}
        h2 {{
            font-size: 20pt;
            margin-top: 0.8em;
            margin-bottom: 0.4em;
            page-break-after: avoid;
        }}
        h3 {{
            font-size: 16pt;
            margin-top: 0.6em;
            margin-bottom: 0.3em;
            page-break-after: avoid;
        }}
        p {{
            margin: 0.5em 0;
            text-align: justify;
        }}
        code {{
            font-family: 'Courier New', 'Consolas', monospace;
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            overflow-x: auto;
            page-break-inside: avoid;
        }}
        pre code {{
            background-color: transparent;
            padding: 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 1em 0;
            page-break-inside: avoid;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        img {{
            max-width: 100%;
            height: auto;
            page-break-inside: avoid;
        }}
        .figure {{
            text-align: center;
            margin: 1em 0;
            page-break-inside: avoid;
        }}
        .figure img {{
            border: 1px solid #ddd;
        }}
        .figure-caption {{
            font-style: italic;
            margin-top: 0.5em;
            font-size: 0.9em;
        }}
        blockquote {{
            border-left: 4px solid #ddd;
            padding-left: 1em;
            margin: 1em 0;
            color: #666;
        }}
        ul, ol {{
            margin: 0.5em 0;
            padding-left: 2em;
        }}
        li {{
            margin: 0.3em 0;
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

print(f"✓ HTML file created: {html_file}")
print("\nTo convert to PDF:")
print("1. Open the HTML file in your browser (Chrome/Edge recommended)")
print("2. Press Ctrl+P (or File > Print)")
print("3. Select 'Save as PDF' as the destination")
print("4. Click 'Save'")
print("\nAlternatively, you can use online converters like:")
print("- https://www.markdowntopdf.com/")
print("- https://cloudconvert.com/md-to-pdf")

