"""Scoring for the Stage 2 accuracy cases.

Every scorer is deterministic and takes no network, so a verdict is reproducible: the only
model involved is the one being tested.

Each `score_*` returns a float in [0, 1] for one case. `edits_wer` and `edits_cer` return
(edits, reference length); the caller divides them to get the case's error rate.
"""

from __future__ import annotations

import json
import re
import string
import unicodedata

# --------------------------------------------------------------------------- edit distance


def _levenshtein(a: list, b: list) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- normalization

_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_CONTRACTIONS = {
    "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
    "cant": "can't", "didnt": "didn't", "doesnt": "doesn't", "wasnt": "wasn't",
}
_PUNCT_RE = re.compile(r"[%s]" % re.escape(string.punctuation))


def normalize_basic(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    return " ".join(text.split())


def normalize_vqa(text: str) -> str:
    """The VQA answer normalization: articles, number words and contractions folded."""
    words = normalize_basic(text).split()
    out = []
    for word in words:
        if word in _ARTICLES:
            continue
        word = _NUMBER_WORDS.get(word, word)
        word = _CONTRACTIONS.get(word, word)
        out.append(word)
    return " ".join(out)


def normalize_asr(text: str, *, language: str = "en") -> str:
    """ASR normalization: NFKC, lowercase, drop punctuation, collapse whitespace.

    Korean keeps its own script; `wer`/`cer` differ in tokenization, not here.
    """
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = re.sub(r"[^\w\s가-힣]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


# --------------------------------------------------------------------------- extraction

_LETTER_ONLY_RE = re.compile(r"^\(?([A-Za-z])\)?[.):,]?$")
_GSM_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_letter(text: str, options: str = "ABCDEFGHIJ") -> str | None:
    """The reply must BE the letter. Nothing is dug out of prose.

    Accepts `B`, `B.`, `(B)`, `B)` and surrounding whitespace or markdown emphasis. Anything
    else -- including a correct letter buried in a sentence -- returns None and scores zero,
    so a reply that ignored the instruction cannot pass on a stray capital.
    """
    if not text:
        return None
    stripped = text.strip().strip("*`_ ").strip()
    match = _LETTER_ONLY_RE.match(stripped)
    if not match:
        return None
    letter = match.group(1).upper()
    return letter if letter in options else None


def extract_gsm8k(text: str) -> str | None:
    """The '#### <n>' final answer, falling back to the last number in the completion."""
    if not text:
        return None
    matches = _GSM_RE.findall(text)
    if matches:
        return matches[-1].replace(",", "")
    matches = _NUMBER_RE.findall(text)
    return matches[-1].replace(",", "") if matches else None


def _as_float(text) -> float | None:
    try:
        return float(str(text).replace(",", "").rstrip("."))
    except (ValueError, AttributeError):
        return None


# --------------------------------------------------------------------------- scorers


def score_mcq(prediction: str, expected: str, options: str = "ABCDEFGHIJ") -> float:
    """1.0 if the extracted letter matches. Refusing to answer scores 0, as it should."""
    return float(extract_letter(prediction, options) == str(expected).strip().upper())


def score_gsm8k(prediction: str, expected: str) -> float:
    got, want = extract_gsm8k(prediction), extract_gsm8k(str(expected))
    if got is None or want is None:
        return 0.0
    a, b = _as_float(got), _as_float(want)
    if a is None or b is None:
        return float(got == want)
    return float(abs(a - b) < 1e-6)


def score_vqa(prediction: str, expected: list[str]) -> float:
    """VQA soft accuracy: min(matching annotators / 3, 1) over the 10 provided answers."""
    got = normalize_vqa(prediction)
    answers = [normalize_vqa(a) for a in (expected if isinstance(expected, list) else [expected])]
    if not answers:
        return 0.0
    matches = sum(1 for a in answers if a == got)
    return min(matches / 3.0, 1.0)


def edits_wer(prediction: str, expected: str) -> tuple[int, int]:
    """(word edits, reference word count)."""
    got = normalize_asr(prediction).split()
    want = normalize_asr(expected).split()
    return _levenshtein(got, want), len(want)


def edits_cer(prediction: str, expected: str) -> tuple[int, int]:
    """(character edits, reference character count), whitespace removed.

    Used for Korean, where word segmentation makes WER close to meaningless -- the same
    convention the FLEURS paper and the Gemma-4 tech report use for ko/ja/zh/th.
    """
    got = list(normalize_asr(prediction).replace(" ", ""))
    want = list(normalize_asr(expected).replace(" ", ""))
    return _levenshtein(got, want), len(want)


# --------------------------------------------------------------------------- IFEval


def _paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _relation(actual: int, relation: str, target: int) -> bool:
    if relation == "less than":
        return actual < target
    if relation == "at least":
        return actual >= target
    return actual == target


def _ifeval_check(instruction_id: str, kwargs: dict, response: str) -> bool | None:
    """One IFEval instruction. Returns None when the instruction is not implemented.

    `score_ifeval` treats None as an error rather than silently scoring the item as a pass.
    """
    kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}
    text = response or ""
    lowered = text.lower()

    if instruction_id == "keywords:existence":
        return all(k.lower() in lowered for k in kwargs.get("keywords", []))
    if instruction_id == "keywords:frequency":
        keyword = str(kwargs.get("keyword", "")).lower()
        count = len(re.findall(re.escape(keyword), lowered))
        return _relation(count, kwargs.get("relation", "at least"), int(kwargs.get("frequency", 1)))
    if instruction_id == "keywords:forbidden_words":
        return not any(k.lower() in lowered for k in kwargs.get("forbidden_words", []))
    if instruction_id == "keywords:letter_frequency":
        letter = str(kwargs.get("letter", "")).lower()
        count = lowered.count(letter)
        return _relation(count, kwargs.get("let_relation", "at least"), int(kwargs.get("let_frequency", 1)))
    if instruction_id == "length_constraints:number_sentences":
        return _relation(len(_sentences(text)), kwargs.get("relation", "at least"), int(kwargs.get("num_sentences", 1)))
    if instruction_id == "length_constraints:number_paragraphs":
        return len(_paragraphs(text)) == int(kwargs.get("num_paragraphs", 1))
    if instruction_id == "length_constraints:number_words":
        return _relation(len(text.split()), kwargs.get("relation", "at least"), int(kwargs.get("num_words", 1)))
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        paras = _paragraphs(text)
        index = int(kwargs.get("nth_paragraph", 1)) - 1
        if not 0 <= index < len(paras) or len(paras) != int(kwargs.get("num_paragraphs", len(paras))):
            return False
        first = paras[index].strip().split()
        return bool(first) and first[0].strip(string.punctuation).lower() == str(kwargs.get("first_word", "")).lower()
    if instruction_id == "detectable_content:number_placeholders":
        return len(re.findall(r"\[[^\[\]]+\]", text)) >= int(kwargs.get("num_placeholders", 1))
    if instruction_id == "detectable_content:postscript":
        marker = str(kwargs.get("postscript_marker", "P.S.")).lower()
        return marker.lower() in lowered
    if instruction_id == "detectable_format:number_bullet_lists":
        return len(re.findall(r"^\s*\*\s+", text, flags=re.MULTILINE)) == int(kwargs.get("num_bullets", 1))
    if instruction_id == "detectable_format:constrained_response":
        options = ["my answer is yes.", "my answer is no.", "my answer is maybe."]
        return any(o in lowered for o in options)
    if instruction_id == "detectable_format:number_highlighted_sections":
        highlights = re.findall(r"\*[^\*\n]+\*", text)
        return len(highlights) >= int(kwargs.get("num_highlights", 1))
    if instruction_id == "detectable_format:multiple_sections":
        splitter = str(kwargs.get("section_spliter", "Section"))
        return len(re.findall(re.escape(splitter), text, flags=re.IGNORECASE)) >= int(kwargs.get("num_sections", 1))
    if instruction_id == "detectable_format:json_format":
        stripped = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
        try:
            json.loads(stripped)
            return True
        except (ValueError, TypeError):
            return False
    if instruction_id == "detectable_format:title":
        return bool(re.search(r"<<[^\n]+>>", text))
    if instruction_id == "combination:two_responses":
        return len(re.split(r"\*\*\*+", text)) == 2
    if instruction_id == "combination:repeat_prompt":
        prompt = str(kwargs.get("prompt_to_repeat", ""))
        return bool(prompt) and text.strip().lower().startswith(prompt.strip().lower())
    if instruction_id == "startend:end_checker":
        phrase = normalize_basic(str(kwargs.get("end_phrase", "")))
        return normalize_basic(text).endswith(phrase) if phrase else False
    if instruction_id == "startend:quotation":
        stripped = text.strip()
        return stripped.startswith('"') and stripped.endswith('"')
    if instruction_id == "change_case:english_capital":
        return text.isupper()
    if instruction_id == "change_case:english_lowercase":
        return text.islower()
    if instruction_id == "change_case:capital_word_frequency":
        count = sum(1 for w in text.split() if w.isupper() and w.isalpha())
        return _relation(count, kwargs.get("capital_relation", "at least"), int(kwargs.get("capital_frequency", 1)))
    if instruction_id == "punctuation:no_comma":
        return "," not in text
    return None


def score_ifeval(prediction: str, expected: dict) -> float:
    """Strict prompt-level accuracy: every instruction on the item must hold."""
    ids = expected.get("instruction_id_list", [])
    all_kwargs = expected.get("kwargs", [])
    if not ids:
        return 0.0
    for index, instruction_id in enumerate(ids):
        kwargs = all_kwargs[index] if index < len(all_kwargs) else {}
        verdict = _ifeval_check(instruction_id, kwargs, prediction)
        if verdict is None:
            raise KeyError(f"IFEval instruction {instruction_id!r} is not implemented")
        if not verdict:
            return 0.0
    return 1.0


# --------------------------------------------------------------------------- dispatch

def score_item(metric: str, prediction: str, expected, options: str = "ABCDEFGHIJ") -> float | tuple[int, int]:
    """Score one case. WER/CER return (edits, reference length)."""
    if metric == "accuracy":
        return score_mcq(prediction, expected, options)
    if metric == "gsm8k":
        return score_gsm8k(prediction, expected)
    if metric == "vqa_accuracy":
        return score_vqa(prediction, expected)
    if metric == "ifeval":
        return score_ifeval(prediction, expected)
    if metric == "wer":
        return edits_wer(prediction, expected)
    if metric == "cer":
        return edits_cer(prediction, expected)
    raise KeyError(f"unknown metric {metric!r}")
