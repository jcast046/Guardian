"""Unit tests for metrics.summarizer module."""

import pytest
from metrics.summarizer import _tok, _bigrams, _fscore


class TestTok:
    """Test suite for metrics.summarizer._tok function.

    Tests text tokenization with various input types and formats.
    """

    def test_tokenize_normal_text(self):
        """Test tokenizing normal text."""
        text = "This is a test sentence."
        result = _tok(text)
        
        assert "this" in result
        assert "is" in result
        assert "a" in result
        assert "test" in result
        assert "sentence" in result

    def test_tokenize_case_insensitive(self):
        """Test tokenization is case insensitive."""
        text = "This IS A Test"
        result = _tok(text)
        
        assert all(token.islower() for token in result)

    def test_tokenize_empty_string(self):
        """Test tokenizing empty string."""
        result = _tok("")
        
        assert result == []

    def test_tokenize_none(self):
        """Test tokenizing None."""
        result = _tok(None)
        
        assert result == []

    def test_tokenize_special_characters(self):
        """Test tokenizing text with special characters."""
        text = "Hello, world! How are you?"
        result = _tok(text)
        
        assert "hello" in result
        assert "world" in result
        assert "how" in result

    def test_tokenize_numbers(self):
        """Test tokenizing text with numbers."""
        text = "Case 123 has 5 items"
        result = _tok(text)
        
        assert "case" in result
        assert "123" in result
        assert "items" in result


class TestBigrams:
    """Test suite for metrics.summarizer._bigrams function.

    Tests bigram generation from token lists with various edge cases.
    """

    def test_bigrams_normal(self):
        """Test generating bigrams from normal text."""
        toks = ["this", "is", "a", "test"]
        result = _bigrams(toks)
        
        assert len(result) == 3
        assert ("this", "is") in result
        assert ("is", "a") in result
        assert ("a", "test") in result

    def test_bigrams_single_token(self):
        """Test generating bigrams from single token."""
        toks = ["test"]
        result = _bigrams(toks)
        
        assert result == []

    def test_bigrams_empty(self):
        """Test generating bigrams from empty list."""
        result = _bigrams([])
        
        assert result == []

    def test_bigrams_two_tokens(self):
        """Test generating bigrams from two tokens."""
        toks = ["hello", "world"]
        result = _bigrams(toks)
        
        assert result == [("hello", "world")]


class TestFscore:
    """Test suite for metrics.summarizer._fscore function.

    Tests F-score calculation between reference and hypothesis token lists
    for ROUGE metric evaluation.
    """

    def test_fscore_perfect_match(self):
        """Test F-score with perfect match."""
        ref = ["this", "is", "a", "test"]
        hyp = ["this", "is", "a", "test"]
        result = _fscore(ref, hyp)
        
        assert result == 1.0

    def test_fscore_no_overlap(self):
        """Test F-score with no overlap."""
        ref = ["this", "is", "a", "test"]
        hyp = ["completely", "different", "words"]
        result = _fscore(ref, hyp)
        
        assert result == 0.0

    def test_fscore_partial_match(self):
        """Test F-score with partial match."""
        ref = ["this", "is", "a", "test"]
        hyp = ["this", "is", "a", "different"]
        result = _fscore(ref, hyp)
        
        assert 0.0 < result < 1.0

    def test_fscore_empty_ref(self):
        """Test F-score with empty reference."""
        ref = []
        hyp = ["this", "is", "a", "test"]
        result = _fscore(ref, hyp)
        
        assert result == 0.0

    def test_fscore_empty_hyp(self):
        """Test F-score with empty hypothesis."""
        ref = ["this", "is", "a", "test"]
        hyp = []
        result = _fscore(ref, hyp)
        
        assert result == 0.0

    def test_fscore_duplicates(self):
        """Test F-score with duplicate tokens."""
        ref = ["this", "is", "this"]
        hyp = ["this", "is", "test"]
        result = _fscore(ref, hyp)
        
        # Should handle duplicates correctly
        assert result > 0.0

    def test_fscore_different_lengths(self):
        """Test F-score with different lengths."""
        ref = ["this", "is", "a", "test", "sentence"]
        hyp = ["this", "test"]
        result = _fscore(ref, hyp)
        
        assert 0.0 < result < 1.0
