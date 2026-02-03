import sys
try:
    import pdfplumber
except ImportError:
    print("请先运行：pip install pdfplumber")
    sys.exit(1)

pdf_path = "DDA3005 Project Report (1).pdf"
txt_path = "report.txt"

with pdfplumber.open(pdf_path) as pdf, open(txt_path, "w", encoding="utf-8") as out:
    for page in pdf.pages:
        text = page.extract_text() or ""
        out.write(text + "\n\n")

print(f"已生成 {txt_path}")

