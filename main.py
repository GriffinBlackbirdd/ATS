import pypdf

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

if __name__ == "__main__":
    pdf_path = "jd.pdf"  # Change to your PDF file name if needed
    content = read_pdf(pdf_path)
    print(content)