import csv
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime, timezone

DATA_DIR = "data"
IN_TEXT_DIR = os.path.join(DATA_DIR, "texts")
IN_META = os.path.join(DATA_DIR, "metadata.csv")

OUT_TEXT_DIR = os.path.join(DATA_DIR, "texts_cleaned")
OUT_META = os.path.join(DATA_DIR, "metadata_clean.csv")

os.makedirs(OUT_TEXT_DIR, exist_ok=True)

RE_ILLUSTRATION = re.compile(r"^\s*\[(illustration|illustrations?)\s*:.*\]\s*$", re.I)

RE_JUNK_LINE = re.compile(
    r"^\s*(?:"
    r"produced by|"
    r"e-text prepared by|"
    r"transcribed from|"
    r"(?:online )?distributed proofreading|"
    r"pgdp\.net|pglaf\.org|"
    r"internet archive|archive\.org|"
    r"this file which includes|"
    r"images of the original pages|"
    r"transcriber's note|"
    r"proofreading team|"
    r"scanned by|"
    r"ocr"
    r")\b",
    re.I
)

RE_GUTENBERG_LICENSE_HINT = re.compile(r"project gutenberg", re.I)

# Contents heading + typical contents line shapes
RE_CONTENTS_HEAD = re.compile(r"^\s*contents\s*$", re.I)
RE_DOTTED_LEADER = re.compile(r"\.{3,}")
RE_PAGE_NUM_LINE = re.compile(r"^\s*\d+\s*$")
RE_CONTENT_ITEM = re.compile(r"^\s{0,6}[a-z0-9\"'(\[]", re.I)  # simple: looks like a title line

# Headings that often mark real start
RE_START_HEADING = re.compile(
    r"^\s*("
    r"preface|introduction|prologue|"
    r"chapter\s+\d+|"
    r"book\s+[ivxlcdm]+|"
    r"part\s+[ivxlcdm]+|"
    r"the\s+[a-z0-9][a-z0-9 ,'\-:;!?]+|"
    r"[ivxlcdm]+\.\s+|"
    r"\d+\.\s+"
    r")\s*$",
    re.I
)

DATA_DIR = "data"
TEXT_DIR = os.path.join(DATA_DIR, "texts_cleaned")
META_CSV = os.path.join(DATA_DIR, "metadata_clean.csv")

OUT_DIR = os.path.join(DATA_DIR, "tokenized")
VOCAB_PATH = os.path.join(DATA_DIR, "vocab.txt")

MAX_VOCAB = 30_000          # total vocab size INCLUDING special tokens below
MIN_FREQ = 2                # ignore tokens that appear < MIN_FREQ times
KEEP_PUNCT_AS_TOKENS = True # recommended for generation

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<num>"]

# Tokenization: words + (optional) punctuation as tokens
# - words: letters/digits/_ and apostrophes inside words
# - punctuation: common sentence punctuation as separate tokens
WORD_RE = r"[a-z]+(?:'[a-z]+)?"
PUNCT_RE = r"[.,!?;:\-—()\[\]\"']"
if KEEP_PUNCT_AS_TOKENS:
    TOKEN_RE = re.compile(rf"{WORD_RE}|{PUNCT_RE}|<num>", re.IGNORECASE)
else:
    TOKEN_RE = re.compile(rf"{WORD_RE}|<num>", re.IGNORECASE)

DIGIT_RE = re.compile(r"\d+")
WHITESPACE_RE = re.compile(r"\s+")

# Optional: normalize curly quotes / dashes to plain forms
CHAR_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"',
    "’": "'", "‘": "'",
    "—": "-", "–": "-",
    "…": "...",
})


@dataclass
class Stats:
    norm_word_count: int
    token_count: int
    unk_count: int

def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def strip_front_matter_lines(lines, max_scan=400):
    """
    Remove obvious boilerplate lines near the beginning (Produced by, PGDP, IA links, etc.)
    and illustration-only lines.
    """
    out = []
    for i, ln in enumerate(lines):
        if i < max_scan:
            if RE_ILLUSTRATION.match(ln):
                continue
            if RE_JUNK_LINE.match(ln):
                continue
            # Many books have a short note about HTML/illustrations; skip those too early on
            if i < 200 and ("note:" in ln.lower() and "project gutenberg" in "\n".join(lines[:200]).lower()):
                # keep conservative: only skip obvious "Note: Project Gutenberg also has an HTML version..."
                if "html version" in ln.lower() or "includes the lovely original illustrations" in ln.lower():
                    continue
        out.append(ln)
    return out

def remove_contents_block(lines, max_scan=500):
    """
    If we find a 'CONTENTS' header early on, remove from that header
    until we hit a likely real heading/story start.
    """
    # Find "CONTENTS"
    idx = None
    for i in range(min(max_scan, len(lines))):
        if RE_CONTENTS_HEAD.match(lines[i].strip()):
            idx = i
            break
    if idx is None:
        return lines

    # Heuristic: remove contents lines after header until a "real" heading appears
    # Real heading: uppercase-ish line or matches start heading
    j = idx + 1
    # skip blank lines immediately after CONTENTS
    while j < len(lines) and lines[j].strip() == "":
        j += 1

    # consume contents entries
    while j < len(lines):
        s = lines[j].strip()
        if s == "":
            j += 1
            continue

        # stop if we encounter a heading
        # or a chapter/preface heading
        if RE_START_HEADING.match(s):
            break

        # stop if line is very "heading-like": mostly uppercase letters/spaces/punct and length > 6
        letters = re.sub(r"[^A-Za-z]", "", s)
        if len(letters) >= 6 and letters.upper() == letters:
            break

        # otherwise continue consuming contents items (including dotted leaders / page numbers)
        j += 1

    # Remove [idx, j) block
    return lines[:idx] + lines[j:]

def find_start_index(lines, max_scan=800):
    """
    Find a good starting point for the real text.
    Prefer the first occurrence of a strong heading (Preface / Chapter / THE ... / I.)
    after we removed obvious boilerplate and contents.
    """
    for i in range(min(max_scan, len(lines))):
        s = lines[i].strip()
        if not s:
            continue
        # skip remaining illustrations
        if RE_ILLUSTRATION.match(s):
            continue
        if RE_START_HEADING.match(s):
            return i
    return 0

def clean_book_text(raw: str) -> str:
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = raw.split("\n")

    lines = strip_front_matter_lines(lines)
    lines = remove_contents_block(lines)

    start = find_start_index(lines)
    lines = lines[start:]

    # Remove long runs of empty lines and trim
    text = "\n".join(lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip() + "\n"
    return text

def count_words_simple(text: str) -> int:
    # similar to your earlier counter: words with apostrophes
    return len(re.findall(r"\b[\w']+\b", text))

def normalize_text(text: str) -> str:
    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # unify common unicode punctuation
    text = text.translate(CHAR_MAP)
    # lowercase
    text = text.lower()
    # numbers -> <num>
    text = DIGIT_RE.sub(" <num> ", text)
    # collapse whitespace
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    # TOKEN_RE finds tokens directly (words/punct/<num>)
    toks = TOKEN_RE.findall(text)
    return toks


def load_metadata_rows() -> List[dict]:
    if not os.path.exists(META_CSV):
        raise FileNotFoundError(f"metadata_clean.csv not found at: {META_CSV}")
    with open(META_CSV, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_metadata_rows(rows: List[dict], fieldnames: List[str]) -> None:
    tmp_path = META_CSV + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp_path, META_CSV)


def build_vocab(counter: Counter) -> Tuple[List[str], Dict[str, int]]:
    # Reserve space for specials
    budget = MAX_VOCAB - len(SPECIAL_TOKENS)
    if budget <= 0:
        raise ValueError("MAX_VOCAB must be larger than number of SPECIAL_TOKENS")

    # Filter by MIN_FREQ and remove any specials if they appear in data
    items = [(tok, c) for tok, c in counter.items() if c >= MIN_FREQ and tok not in SPECIAL_TOKENS]

    # Sort by frequency then token to stabilize
    items.sort(key=lambda x: (-x[1], x[0]))
    vocab_tokens = SPECIAL_TOKENS + [tok for tok, _ in items[:budget]]

    tok2id = {tok: i for i, tok in enumerate(vocab_tokens)}
    return vocab_tokens, tok2id


def write_vocab(vocab_tokens: List[str]) -> None:
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        for tok in vocab_tokens:
            f.write(tok + "\n")


def tokenize_and_save_ids(ebook_id: str, toks: List[str], tok2id: Dict[str, int]) -> Stats:
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{ebook_id}.ids.txt")

    unk_id = tok2id["<unk>"]
    ids = []
    unk = 0
    for t in toks:
        i = tok2id.get(t, unk_id)
        if i == unk_id and t != "<unk>":
            unk += 1
        ids.append(i)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, ids)))

    # norm_word_count is a “word-ish” count (excluding punctuation if present)
    # For your tracking, token_count is the real number of tokens used for training.
    norm_word_count = sum(1 for t in toks if re.fullmatch(WORD_RE, t))
    return Stats(norm_word_count=norm_word_count, token_count=len(ids), unk_count=unk)


def main():
    if not os.path.exists(IN_META):
        raise FileNotFoundError(f"Missing {IN_META}")

    with open(IN_META, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for r in rows:
        eid = r.get("ebook_id")
        in_path = r.get("txt_path", "")
        if not eid or not in_path or not os.path.exists(in_path):
            continue

        with open(in_path, "r", encoding="utf-8") as f:
            raw = f.read()

        cleaned = clean_book_text(raw)

        out_path = os.path.join(OUT_TEXT_DIR, f"{eid}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        r2 = dict(r)
        r2["txt_path"] = out_path
        r2["cleaned_at_utc"] = utc_now_iso()
        r2["clean_word_count"] = str(count_words_simple(cleaned))
        out_rows.append(r2)

    # write new metadata
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    if "cleaned_at_utc" not in fieldnames:
        fieldnames.append("cleaned_at_utc")
    if "clean_word_count" not in fieldnames:
        fieldnames.append("clean_word_count")

    with open(OUT_META, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Cleaned {len(out_rows)} books")
    print(f"Texts -> {OUT_TEXT_DIR}/")
    print(f"Metadata -> {OUT_META}")


    rows = load_metadata_rows()

    # Pass 1: build global token frequency counter from ALL books
    counter = Counter()
    total_tokens_seen = 0

    for r in rows:
        ebook_id = r["ebook_id"]
        txt_path = r["txt_path"]
        if not txt_path or not os.path.exists(txt_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            raw = f.read()

        norm = normalize_text(raw)
        toks = tokenize(norm)

        counter.update(toks)
        total_tokens_seen += len(toks)

    print(f"[pass1] scanned {len(rows)} metadata rows")
    print(f"[pass1] total tokens observed (pre-vocab): {total_tokens_seen:,}")
    print(f"[pass1] unique tokens observed: {len(counter):,}")

    vocab_tokens, tok2id = build_vocab(counter)
    write_vocab(vocab_tokens)

    print(f"[vocab] written: {VOCAB_PATH}")
    print(f"[vocab] size: {len(vocab_tokens):,} (MAX_VOCAB={MAX_VOCAB:,}, MIN_FREQ={MIN_FREQ})")

    # Pass 2: tokenize each book into ids + collect stats + update metadata.csv
    total_token_count = 0
    total_unk = 0

    # Add new columns if not present
    extra_cols = ["norm_word_count", "token_count", "unk_count", "unk_rate"]
    fieldnames = list(rows[0].keys()) if rows else []
    for c in extra_cols:
        if c not in fieldnames:
            fieldnames.append(c)

    for r in rows:
        ebook_id = r["ebook_id"]
        txt_path = r["txt_path"]
        if not txt_path or not os.path.exists(txt_path):
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            raw = f.read()

        norm = normalize_text(raw)
        toks = tokenize(norm)

        # Optionally wrap each book with BOS/EOS for training
        toks = ["<bos>"] + toks + ["<eos>"]

        stats = tokenize_and_save_ids(ebook_id, toks, tok2id)

        r["norm_word_count"] = str(stats.norm_word_count)
        r["token_count"] = str(stats.token_count)
        r["unk_count"] = str(stats.unk_count)
        r["unk_rate"] = f"{(stats.unk_count / max(1, stats.token_count)):.6f}"

        total_token_count += stats.token_count
        total_unk += stats.unk_count

    save_metadata_rows(rows, fieldnames)

    print(f"[pass2] tokenized books -> {OUT_DIR}/<ebook_id>.ids.txt")
    print(f"[pass2] updated metadata -> {META_CSV}")
    print(f"[totals] total training tokens (word-level): {total_token_count:,}")
    print(f"[totals] total UNK tokens: {total_unk:,}  ({total_unk / max(1, total_token_count):.4%})")


if __name__ == "__main__":
    main()