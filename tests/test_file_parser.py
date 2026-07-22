"""Tests for file_parser.py"""
from src.ingest.file_parser import parse_text


def test_parse_txt():
    result = parse_text("test.txt", b"hello world")
    assert "hello world" in result


def test_parse_pdf_simple():
    # minimal valid PDF
    pdf_bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Resources<<>>>>endobj\nxref\n0 4\n...\n"
    result = parse_text("test.pdf", pdf_bytes)
    assert isinstance(result, str)
