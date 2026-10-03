"""Focused exact byte-BPE pre-tokenizer tests."""

from src.wide_bio_tokenizer import byte_alphabet, pre_tokenize


# ##################################################################
# qwen35 pre-tokenizer
# retain every source character in exactly one qwen35 pre-tokenizer piece before byte BPE merges.
def test_qwen35_pre_tokenizer_tiles_unicode_and_whitespace() -> None:
    text = "Ren's  tall,\n\n  house 42! Zoë"
    pieces = pre_tokenize(text)
    assert "".join(pieces) == text
    assert pieces == ["Ren", "'s", " ", " tall", ",\n\n", " ", " house", " ", "4", "2", "!", " Zoë"]


# ##################################################################
# GPT2 byte alphabet
# assert the actual bijection used for every vocabulary byte and Unicode source byte.
def test_gpt2_byte_alphabet_round_trips_all_bytes() -> None:
    encoded, decoded = byte_alphabet()
    assert len(encoded) == len(decoded) == 256
    assert bytes(decoded[encoded[value]] for value in range(256)) == bytes(range(256))
