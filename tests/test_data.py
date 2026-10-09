import pytest
from src.data_cleaning import clean_imdb_text

def test_clean_text():
    html_text = "<br>Hello! <b>World</b>"
    # Should strip HTML tags
    assert clean_imdb_text(html_text) == "Hello! World"

    messy_text = "   This   is   a    test!!!!  "
    # Should collapse whitespaces
    assert clean_imdb_text(messy_text) == "This is a test!!!!"
