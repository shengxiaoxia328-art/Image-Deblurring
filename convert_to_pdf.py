"""
Convert Markdown report to PDF
"""
import sys
import os

def convert_markdown_to_pdf(md_file, pdf_file):
    """Convert markdown to PDF using available tools"""
    
    # Try method 1: markdown2pdf (requires markdown-pdf)
    try:
        import subprocess
        result = subprocess.run(
            ['npx', '--yes', 'markdown-pdf', md_file, '-o', pdf_file],
            capture_output=True,
            text=True,
            timeout=120
        )
        if result.returncode == 0:
            print(f"Successfully converted using markdown-pdf: {pdf_file}")
            return True
    except Exception as e:
        print(f"markdown-pdf method failed: {e}")
    
    # Try method 2: pypandoc (requires pandoc)
    try:
        import pypandoc
        pypandoc.convert_file(
            md_file,
            'pdf',
            outputfile=pdf_file,
            extra_args=['--pdf-engine=xelatex', '--variable=geometry:margin=1in']
        )
        print(f"Successfully converted using pypandoc: {pdf_file}")
        return True
    except ImportError:
        print("pypandoc not available. Install with: pip install pypandoc")
    except Exception as e:
        print(f"pypandoc conversion failed: {e}")
    
    # Try method 3: markdown + weasyprint
    try:
        import markdown
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Add basic styling
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Times New Roman', serif; margin: 2cm; line-height: 1.6; }}
                h1 {{ font-size: 24pt; margin-top: 1em; margin-bottom: 0.5em; }}
                h2 {{ font-size: 20pt; margin-top: 0.8em; margin-bottom: 0.4em; }}
                h3 {{ font-size: 16pt; margin-top: 0.6em; margin-bottom: 0.3em; }}
                p {{ margin: 0.5em 0; text-align: justify; }}
                code {{ font-family: 'Courier New', monospace; background-color: #f4f4f4; padding: 2px 4px; }}
                pre {{ background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd; overflow-x: auto; }}
                table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        HTML(string=html_doc).write_pdf(pdf_file)
        print(f"Successfully converted using weasyprint: {pdf_file}")
        return True
    except ImportError as e:
        print(f"weasyprint not available: {e}")
        print("Install with: pip install markdown weasyprint")
    except Exception as e:
        print(f"weasyprint conversion failed: {e}")
    
    # Try method 4: markdown + pdfkit (requires wkhtmltopdf)
    try:
        import markdown
        import pdfkit
        
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Times New Roman', serif; margin: 2cm; line-height: 1.6; }}
                h1 {{ font-size: 24pt; margin-top: 1em; margin-bottom: 0.5em; }}
                h2 {{ font-size: 20pt; margin-top: 0.8em; margin-bottom: 0.4em; }}
                h3 {{ font-size: 16pt; margin-top: 0.6em; margin-bottom: 0.3em; }}
                p {{ margin: 0.5em 0; text-align: justify; }}
                code {{ font-family: 'Courier New', monospace; background-color: #f4f4f4; padding: 2px 4px; }}
                pre {{ background-color: #f4f4f4; padding: 10px; border: 1px solid #ddd; overflow-x: auto; }}
                table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        pdfkit.from_string(html_doc, pdf_file)
        print(f"Successfully converted using pdfkit: {pdf_file}")
        return True
    except ImportError:
        print("pdfkit not available. Install with: pip install pdfkit")
        print("Also need: wkhtmltopdf (https://wkhtmltopdf.org/downloads.html)")
    except Exception as e:
        print(f"pdfkit conversion failed: {e}")
    
    return False

if __name__ == "__main__":
    md_file = "DDA3005_Final_Report.md"
    pdf_file = "DDA3005_Final_Report.pdf"
    
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found!")
        sys.exit(1)
    
    print(f"Converting {md_file} to {pdf_file}...")
    print("Trying different conversion methods...\n")
    
    if convert_markdown_to_pdf(md_file, pdf_file):
        print(f"\n✓ Conversion successful! PDF saved as: {pdf_file}")
    else:
        print("\n✗ All conversion methods failed.")
        print("\nPlease install one of the following:")
        print("1. markdown + weasyprint: pip install markdown weasyprint")
        print("2. pypandoc + pandoc: pip install pypandoc (and install pandoc separately)")
        print("3. pdfkit + wkhtmltopdf: pip install pdfkit (and install wkhtmltopdf separately)")
        print("4. Or use online converters like https://www.markdowntopdf.com/")

