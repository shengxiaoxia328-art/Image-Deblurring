import sys
try:
    import pdfplumber
    with pdfplumber.open('Instruction.pdf') as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() or ''
            text += '\n\n'
        print(text)
except ImportError:
    try:
        import PyPDF2
        with open('Instruction.pdf', 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            print(text)
    except ImportError:
        print("需要安装 pdfplumber 或 PyPDF2: pip install pdfplumber")
        sys.exit(1)

