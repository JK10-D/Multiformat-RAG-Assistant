# -*- coding: utf-8 -*-

import os
import PyPDF2
import pandas as pd
import requests
from bs4 import BeautifulSoup


def load_pdf(filepath):
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


def load_csv(filepath):
    df = pd.read_csv(filepath)
    rows = []
    for _, row in df.iterrows():
        row_text = " | ".join([f"{col}: {val}" for col, val in row.items()])
        rows.append(row_text)
    return "\n".join(rows)


def load_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(url, timeout=10, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def load_document(source):
    if isinstance(source, str) and source.startswith("http"):
        return load_url(source)
    ext = os.path.splitext(str(source))[1].lower()
    if ext == ".pdf":
        return load_pdf(source)
    elif ext == ".csv":
        return load_csv(source)
    else:
        raise ValueError(f"Format non supporté : {ext}")