from pathlib import Path
from unstructured.partition.pdf import partition_pdf
import pdfplumber
from tabulate import tabulate

TEMP_DIR = Path("./temp")

def extract_tables(pdf_path: Path) -> str:
    tables_md = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            for t_idx, table in enumerate(page.extract_tables(), start=1):
                if not table or len(table) < 2:
                    continue
                tables_md.append(
                    f"#### Table {t_idx} (page {page_idx})\n"
                    + tabulate(table, headers="firstrow", tablefmt="github")
                )
    return "\n\n".join(tables_md)

def pdf_to_md(pdf_path: Path):
    print(f"Processing: {pdf_path.name}")

    # Poppler-free extraction
    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="fast",                 # IMPORTANT
        infer_table_structure=False      # tables handled separately
    )

    body_md = "\n\n".join(str(e) for e in elements)

    tables_md = extract_tables(pdf_path)

    md_parts = [
        f"# {pdf_path.stem}",
        "",
        "> Automatically extracted for LLM / agent consumption.",
        "",
        body_md
    ]

    if tables_md.strip():
        md_parts.extend([
            "",
            "---",
            "",
            "## Extracted Tables",
            "",
            tables_md
        ])

    pdf_path.with_suffix(".md").write_text(
        "\n".join(md_parts), encoding="utf-8"
    )

def main():
    pdfs = list(TEMP_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in ./temp")
        return

    for pdf in pdfs:
        try:
            pdf_to_md(pdf)
        except Exception as e:
            print(f"Failed on {pdf.name}: {e}")

if __name__ == "__main__":
    main()
