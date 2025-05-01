import pdfplumber
import pandas as pd

import re
import string
from nltk.corpus import words

# Preload dictionary
word_list = set(words.words())

def decode_cid_string(text):
    """Decode strings with cid-style encoding like (cid:72) to characters."""
    def cid_to_char(match):
        num = int(match.group(1))
        try:
            return chr(num)
        except:
            return ''
    text = re.sub(r"\(cid:(\d+)\)", cid_to_char, text)

    def caesar_decrypt(text, shift):
        def shift_char(c):
            if c in string.ascii_uppercase:
                return chr((ord(c) - 65 - shift) % 26 + 65)
            elif c in string.ascii_lowercase:
                return chr((ord(c) - 97 - shift) % 26 + 97)
            return c
        return ''.join(shift_char(c) for c in text)

    def best_caesar_decrypt(text):
        best_score = 0
        best_result = text
        for shift in range(1, 26):
            candidate = caesar_decrypt(text, shift)
            # print('candidate value is ', candidate)
            score = sum(1 for w in candidate.split() if w.lower() in word_list)
            # print('score is ', score)
            if score > best_score:
                best_score = score
                best_result = candidate
        return best_result

    # Decode Caesar only if it looks like gibberish
    text = best_caesar_decrypt(text)

    return text

def decode_if_needed(cell):
    if not isinstance(cell, str):
        return cell
    if "(cid:" in cell and "(cid:42)(cid:72)(cid:68)(cid:85)(cid:3)(cid:54)(cid:75)(cid:76)(cid:73)(cid:87)(cid:3)(cid:44)(cid:81)(cid:71)(cid:76)(cid:70)(cid:68)(cid:87)(cid:82)(cid:85)" in cell:
        print("cell value " + cell)
        cell = decode_cid_string(cell)
    return cell

def clean_table_data(df):
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace(
        {"—": "No", "(cid:859)": "Yes", "": "N/A", None: "N/A"}, 
        inplace=True
    )
    df.dropna(how="all", inplace=True)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    return df.applymap(decode_if_needed)

def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            for table in page_tables:
                if table and len(table) > 1:
                    header = table[0]
                    if not header or any(h is None or h.strip() == "" for h in header):
                        print(f"[INFO] Skipped malformed table on page {page.page_number}")
                        continue
                    df = pd.DataFrame(table[1:], columns=header)
                    df = clean_table_data(df)
                    df.attrs['page_number'] = page.page_number  # optional traceability
                    tables.append(df)
    print(f"[INFO] Extracted {len(tables)} clean tables from {pdf_path}")
    return tables
