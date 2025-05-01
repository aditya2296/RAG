from data_extraction.extractor import extract_text, extract_tables

pdf_path = "swift.pdf"

# Extract text
text_data = extract_text(pdf_path)

# Extract tables
tables = extract_tables(pdf_path)

from pandasgui import show
show(*tables)