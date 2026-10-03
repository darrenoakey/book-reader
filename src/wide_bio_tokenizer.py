"""Exact local token counter rebuilt from a pinned Ollama `/api/show` capture, failing closed whenever exactness is unprovable.

Ollama has no tokenize endpoint, but `/api/show` with verbose=true publishes the model's complete byte-level BPE tokenizer
(tokens, merges, token types, pre-tokenizer name). This module rebuilds that tokenizer in pure Python and refuses, with a
typed TokenizerRefusal, anything whose token count could differ from the server's: an unpinned or altered capture, an
unknown pre-tokenizer, an inconsistent vocabulary, text that is not NFC (llama.cpp does not normalise, HF does), text that
contains a control/user-defined special-token string (the server would parse it as a single special token), or a lossy
round trip. It never contacts a model and never calls inference.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

CAPTURE_SCHEMA = 1
SUPPORTED_PRE = frozenset({"qwen35"})
CONTROL_TOKEN_TYPES = frozenset({3, 4})
# U+017F (long s) case-folds to "s", so a case-insensitive engine may read it as part of a contraction while a byte-exact one does not; the two cannot be proven equal, so it is refused.
AMBIGUOUS_CASEFOLD = frozenset({"\u017f"})
WHITESPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000"
)
CONTRACTION_TAILS = ("s", "t", "re", "ve", "m", "ll", "d")
SHOW_KEYS = (
    "template",
    "details",
    "capabilities",
    "parameters",
    "modelfile",
    "requires",
)


class TokenizerRefusal(Exception):
    """The token count cannot be proven exact; callers must stop rather than estimate."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


# ##################################################################
# capture digest
# sha256 of the exact capture file bytes; this is the pin that binds a proof run to one tokenizer.
def capture_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ##################################################################
# build capture
# reduces a verbose `/api/show` response to the tokenizer-bearing metadata plus the identity it was read for (model and endpoint), so the pinned hash covers which model the tokenizer belongs to.
def build_capture(model: str, url: str, show: dict) -> dict:
    info = show.get("model_info")
    if not isinstance(info, dict):
        raise TokenizerRefusal("show response has no model_info", "capture_malformed")
    kept = {
        key: value
        for key, value in info.items()
        if key.startswith("tokenizer.ggml.") and key != "tokenizer.ggml.scores"
    }
    kept.update(
        {
            key: value
            for key, value in info.items()
            if key.endswith(".context_length") or key == "general.architecture"
        }
    )
    return {
        "schema": CAPTURE_SCHEMA,
        "model": model,
        "url": url,
        "model_info": kept,
        **{key: show[key] for key in SHOW_KEYS if key in show},
    }


# ##################################################################
# byte alphabet
# the GPT-2 byte <-> printable-unicode bijection used by every byte-level BPE vocabulary.
def byte_alphabet() -> tuple[dict[int, str], dict[str, int]]:
    keep = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping, extra = {}, 0
    for byte in range(256):
        if byte in keep:
            mapping[byte] = chr(byte)
        else:
            mapping[byte] = chr(256 + extra)
            extra += 1
    return mapping, {char: byte for byte, char in mapping.items()}


def category(char: str) -> str:
    return unicodedata.category(char)


def letter(char: str) -> bool:
    return category(char)[0] == "L"


def mark(char: str) -> bool:
    return category(char)[0] == "M"


def number(char: str) -> bool:
    return category(char)[0] == "N"


def letter_or_mark(char: str) -> bool:
    return category(char)[0] in "LM"


# ##################################################################
# contraction length
# length of a case-insensitive English contraction ('s 't 're 've 'm 'll 'd) starting at position, else 0.
def contraction_length(text: str, position: int) -> int:
    if text[position] != "'":
        return 0
    for tail in CONTRACTION_TAILS:
        if text[position + 1 : position + 1 + len(tail)].lower() == tail:
            return 1 + len(tail)
    return 0


# ##################################################################
# pre-tokenize
# the qwen35 split pattern `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+` as an explicit scanner with the regex's alternation order and backtracking, covering the text with consecutive pieces.
def pre_tokenize(text: str) -> list[str]:
    pieces: list[str] = []
    size, position = len(text), 0
    while position < size:
        char = text[position]
        end = 0
        length = contraction_length(text, position)
        if length:
            end = position + length
        elif (
            char not in "\r\n"
            and not letter(char)
            and not number(char)
            and position + 1 < size
            and letter_or_mark(text[position + 1])
        ):
            end = position + 2
            while end < size and letter_or_mark(text[end]):
                end += 1
        elif letter_or_mark(char):
            end = position + 1
            while end < size and letter_or_mark(text[end]):
                end += 1
        elif number(char):
            end = position + 1
        else:
            start = (
                position + 1
                if char == " " and position + 1 < size and symbol(text[position + 1])
                else position
            )
            if start < size and symbol(text[start]):
                end = start + 1
                while end < size and symbol(text[end]):
                    end += 1
                while end < size and text[end] in "\r\n":
                    end += 1
        if not end:
            run = position
            while run < size and text[run] in WHITESPACE:
                run += 1
            last = max(
                (index for index in range(position, run) if text[index] in "\r\n"),
                default=-1,
            )
            if last >= 0:
                end = last + 1
            elif run == size:
                end = run
            elif run - position > 1:
                end = run - 1
            else:
                end = run
        pieces.append(text[position:end])
        position = end
    return pieces


# ##################################################################
# symbol
# a character the punctuation alternative may consume: not whitespace, letter, mark, or number.
def symbol(char: str) -> bool:
    return char not in WHITESPACE and category(char)[0] not in "LMN"


# ##################################################################
# exact tokenizer
# byte-level BPE built from the capture; ids are positions in the published token list.
class ExactTokenizer:
    def __init__(self, capture: dict, capture_sha256: str) -> None:
        info = capture["model_info"]
        self.capture_sha256 = capture_sha256
        self.model = capture["model"]
        self.pre = info["tokenizer.ggml.pre"]
        tokens, types, merges = (
            info["tokenizer.ggml.tokens"],
            info["tokenizer.ggml.token_type"],
            info["tokenizer.ggml.merges"],
        )
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.tokens = tokens
        contexts = [
            value for key, value in info.items() if key.endswith(".context_length")
        ]
        self.context_length = max(
            (value for value in contexts if isinstance(value, int)), default=0
        )
        self.ranks: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(merges):
            left, right = merge.split(" ")
            self.ranks.setdefault((left, right), rank)
        self.byte_to_char, self.char_to_byte = byte_alphabet()
        specials = [
            token
            for token, kind in zip(tokens, types)
            if kind in CONTROL_TOKEN_TYPES and token
        ]
        self.special_pattern = (
            re.compile(
                "|".join(
                    re.escape(token)
                    for token in sorted(specials, key=len, reverse=True)
                )
            )
            if specials
            else None
        )
        self.specials = tuple(specials)
        self.cache: dict[str, tuple[int, ...]] = {}

    def merge(self, word: str) -> tuple[int, ...]:
        symbols = list(word)
        while len(symbols) > 1:
            best = min(
                (
                    (self.ranks.get(pair, -1), index)
                    for index, pair in enumerate(itertools.pairwise(symbols))
                ),
                key=lambda item: (item[0] < 0, item[0], item[1]),
            )
            if best[0] < 0:
                break
            pair = (symbols[best[1]], symbols[best[1] + 1])
            merged: list[str] = []
            index = 0
            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and (symbols[index], symbols[index + 1]) == pair
                ):
                    merged.append(symbols[index] + symbols[index + 1])
                    index += 2
                else:
                    merged.append(symbols[index])
                    index += 1
            symbols = merged
        return tuple(self.vocab[symbol_text] for symbol_text in symbols)

    def check_text(self, text: str) -> None:
        if unicodedata.normalize("NFC", text) != text:
            raise TokenizerRefusal(
                "text is not NFC; the server and a normalising tokenizer would disagree",
                "text_not_nfc",
            )
        if self.special_pattern is not None:
            hit = self.special_pattern.search(text)
            if hit:
                raise TokenizerRefusal(
                    f"text contains special token {hit.group(0)!r}",
                    "text_contains_special_token",
                )
        if any(char in text for char in AMBIGUOUS_CASEFOLD):
            raise TokenizerRefusal(
                "text contains a case-fold ambiguous character",
                "text_casefold_ambiguous",
            )
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as error:
            raise TokenizerRefusal(
                "text is not valid unicode", "text_not_utf8"
            ) from error

    def encode(self, text: str) -> list[int]:
        self.check_text(text)
        ids: list[int] = []
        for piece in pre_tokenize(text):
            word = "".join(self.byte_to_char[byte] for byte in piece.encode("utf-8"))
            if word not in self.cache:
                self.cache[word] = self.merge(word)
            ids.extend(self.cache[word])
        if self.decode(ids) != text:
            raise TokenizerRefusal("token round trip is lossy", "round_trip_failed")
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        data = bytearray()
        for token_id in ids:
            data.extend(self.char_to_byte[char] for char in self.tokens[token_id])
        return data.decode("utf-8")

    def count(self, text: str) -> int:
        return len(self.encode(text))


# ##################################################################
# validate capture
# every structural fact the reconstruction depends on; any gap is a refusal, never a best effort.
def validate_capture(capture: dict, model: str) -> None:
    if not isinstance(capture, dict) or capture.get("schema") != CAPTURE_SCHEMA:
        raise TokenizerRefusal("capture schema is not supported", "capture_malformed")
    if capture.get("model") != model:
        raise TokenizerRefusal(
            f"capture is for {capture.get('model')!r}, not {model!r}",
            "capture_model_mismatch",
        )
    info = capture.get("model_info")
    if not isinstance(info, dict):
        raise TokenizerRefusal("capture has no model_info", "capture_malformed")
    if info.get("tokenizer.ggml.model") != "gpt2":
        raise TokenizerRefusal(
            "tokenizer is not byte-level BPE (gpt2)", "tokenizer_unsupported"
        )
    if info.get("tokenizer.ggml.pre") not in SUPPORTED_PRE:
        raise TokenizerRefusal(
            f"pre-tokenizer {info.get('tokenizer.ggml.pre')!r} is not proven",
            "pre_tokenizer_unproven",
        )
    tokens, types, merges = (
        info.get("tokenizer.ggml.tokens"),
        info.get("tokenizer.ggml.token_type"),
        info.get("tokenizer.ggml.merges"),
    )
    if not (
        isinstance(tokens, list)
        and isinstance(types, list)
        and isinstance(merges, list)
    ):
        raise TokenizerRefusal(
            "capture lacks tokens, token types or merges", "capture_malformed"
        )
    if (
        len(tokens) != len(types)
        or not tokens
        or not all(isinstance(token, str) for token in tokens)
    ):
        raise TokenizerRefusal(
            "token and token type tables disagree", "vocab_inconsistent"
        )
    vocab = set(tokens)
    if len(vocab) != len(tokens):
        raise TokenizerRefusal("vocabulary has duplicate tokens", "vocab_inconsistent")
    byte_to_char, _ = byte_alphabet()
    if any(char not in vocab for char in byte_to_char.values()):
        raise TokenizerRefusal(
            "vocabulary lacks a byte-level base token", "vocab_inconsistent"
        )
    for merge in merges:
        parts = merge.split(" ") if isinstance(merge, str) else []
        if (
            len(parts) != 2
            or parts[0] not in vocab
            or parts[1] not in vocab
            or parts[0] + parts[1] not in vocab
        ):
            raise TokenizerRefusal(
                f"merge {merge!r} does not resolve inside the vocabulary",
                "vocab_inconsistent",
            )


# ##################################################################
# load exact tokenizer
# the only way to obtain a counter: the capture must hash to the pin, name the configured model, and validate completely.
def load_exact_tokenizer(path: Path, pinned_sha256: str, model: str) -> ExactTokenizer:
    if not path.is_file():
        raise TokenizerRefusal(
            f"tokenizer capture {path} does not exist", "capture_missing"
        )
    digest = capture_digest(path)
    if digest != pinned_sha256:
        raise TokenizerRefusal(
            "tokenizer capture does not match its pinned sha256",
            "capture_hash_mismatch",
        )
    try:
        capture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TokenizerRefusal(
            "tokenizer capture is not readable JSON", "capture_malformed"
        ) from error
    validate_capture(capture, model)
    return ExactTokenizer(capture, digest)
