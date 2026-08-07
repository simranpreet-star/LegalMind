"""
Extracts candidate clauses from a contract PDF.

Primary strategy is numbered-header segmentation. The fallback for documents
without numbered headers is regex sentence splitting rather than spaCy: the
deployed app runs on a container where downloading en_core_web_sm at first
request is both slow and a failure mode that only shows up in production.
spaCy is still supported if a pipeline is passed in explicitly, so local
analysis can use the better splitter.
"""

import re

HEADER_PATTERN = re.compile(
    r"^\s*(?:(?:ARTICLE|SECTION)\s+[IVXLC\d]+\.?|(?:\d{1,2}(?:\.\d{1,2}){0,2})\.?)\s+\S",
    re.IGNORECASE,
)

# Split after . ! ? when followed by whitespace and a capital/digit, but not
# after common legal abbreviations that would otherwise fragment a sentence.
_ABBREVIATIONS = r"(?<!\bNo)(?<!\bInc)(?<!\bLtd)(?<!\bCo)(?<!\bCorp)(?<!\bv)(?<!\bvs)(?<!\bSec)(?<!\bArt)"
SENTENCE_SPLIT = re.compile(rf"{_ABBREVIATIONS}(?<=[.!?])\s+(?=[A-Z0-9])")

MIN_CLAUSE_CHARS = 20


def extract_text_from_pdf(path):
    import pdfplumber

    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def segment_by_numbered_headers(text):
    clauses = []
    current_header = None
    current_lines = []

    for line in text.split("\n"):
        if HEADER_PATTERN.match(line):
            if current_header is not None or current_lines:
                clauses.append(
                    {"header": current_header, "text": "\n".join(current_lines).strip()}
                )
            current_header = line.strip()
            current_lines = [line.strip()]
        else:
            current_lines.append(line)

    if current_header is not None or current_lines:
        clauses.append({"header": current_header, "text": "\n".join(current_lines).strip()})

    return [c for c in clauses if len(c["text"]) > MIN_CLAUSE_CHARS]


def segment_by_sentences(text, nlp=None, max_sentences_per_clause=4):
    """
    Groups sentences into fixed-size chunks. Pass a loaded spaCy pipeline as
    `nlp` for better boundary detection; otherwise a regex splitter is used.
    """
    if nlp is not None:
        sentences = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    else:
        sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]

    clauses = []
    for i in range(0, len(sentences), max_sentences_per_clause):
        chunk = " ".join(sentences[i:i + max_sentences_per_clause])
        if len(chunk) > MIN_CLAUSE_CHARS:
            clauses.append({"header": None, "text": chunk})
    return clauses


def parse_contract_pdf(path, nlp=None, min_headers_expected=2):
    text = extract_text_from_pdf(path)
    clauses = segment_by_numbered_headers(text)
    if len(clauses) < min_headers_expected:
        clauses = segment_by_sentences(text, nlp=nlp)
    return clauses


def parse_contract_text(text, nlp=None, min_headers_expected=2):
    """Same segmentation for text pasted directly into the app."""
    clauses = segment_by_numbered_headers(text)
    if len(clauses) < min_headers_expected:
        clauses = segment_by_sentences(text, nlp=nlp)
    return clauses
