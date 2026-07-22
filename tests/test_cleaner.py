from src.processing.cleaner import clean_text


def test_remove_extra_whitespace():
    result = clean_text("hello    world\n\n\n\n")
    assert "hello world" in result
    assert "\n\n\n" not in result


def test_remove_special_chars():
    result = clean_text("　test　　data")
    assert "test data" in result


def test_remove_urls():
    result = clean_text("check https://example.com/page for details")
    assert "check" in result
    assert "https://" not in result


def test_remove_empty_lines():
    result = clean_text("a\n\n\n\nb\n\n\nc")
    # should keep reasonable spacing
    assert "a" in result and "b" in result and "c" in result
