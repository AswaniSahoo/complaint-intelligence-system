"""Tests for src/preprocess.py | text cleaning and data loading."""

import pytest
import pandas as pd
from src.preprocess import clean_text


class TestCleanText:
    """Unit tests for the clean_text function."""

    def test_basic_cleaning(self):
        assert clean_text("Hello World!") == "hello world!"

    def test_removes_urls(self):
        result = clean_text("Visit http://example.com for info")
        assert "http" not in result
        assert "example.com" not in result

    def test_removes_emails(self):
        result = clean_text("Contact support@bank.com immediately")
        assert "support@bank.com" not in result

    def test_removes_special_characters(self):
        result = clean_text("Price is $100 & tax #included")
        assert "$" not in result
        assert "#" not in result
        assert "&" not in result

    def test_preserves_basic_punctuation(self):
        result = clean_text("Hello, world. How are you?")
        assert "," in result
        assert "." in result
        assert "?" in result

    def test_collapses_whitespace(self):
        result = clean_text("too   many    spaces   here")
        assert "  " not in result

    def test_handles_nan(self):
        assert clean_text(float("nan")) == ""

    def test_handles_none_via_nan(self):
        assert clean_text(pd.NA) == ""

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_handles_numeric_input(self):
        result = clean_text(12345)
        assert result == "12345"

    def test_lowercases(self):
        assert clean_text("UPPERCASE TEXT") == "uppercase text"

    def test_very_long_text(self):
        long_text = "complaint " * 10000
        result = clean_text(long_text)
        assert len(result) > 0
        assert result == ("complaint " * 10000).strip()

    def test_unicode_characters(self):
        result = clean_text("café résumé naïve")
        # Unicode accented chars get stripped by the regex [^a-z0-9...]
        assert isinstance(result, str)
        assert len(result) > 0

    def test_mixed_content(self):
        text = "Contact me at test@email.com or visit https://bank.com. My acc# is 12345!"
        result = clean_text(text)
        assert "@" not in result
        assert "https" not in result
        assert "12345" in result
