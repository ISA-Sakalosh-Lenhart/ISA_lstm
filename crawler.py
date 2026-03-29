import csv
import hashlib
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BOOKSHELF_PAGES = [
    "https://www.gutenberg.org/ebooks/bookshelf/216",
    "https://www.gutenberg.org/ebooks/bookshelf/216?start_index=26",
    "https://www.gutenberg.org/ebooks/bookshelf/216?start_index=51",
    "https://www.gutenberg.org/ebooks/bookshelf/18",
    "https://www.gutenberg.org/ebooks/bookshelf/198",
    "https://www.gutenberg.org/ebooks/bookshelf/37",
    "https://www.gutenberg.org/ebooks/bookshelf/52",
    "https://www.gutenberg.org/ebooks/bookshelf/18",
    "https://www.gutenberg.org/ebooks/bookshelf/213",
    "https://www.gutenberg.org/ebooks/bookshelf/20",
]

DATA_DIR = "data"
TEXT_DIR = os.path.join(DATA_DIR, "texts")
META_CSV = os.path.join(DATA_DIR, "metadata.csv")

REQUEST_SLEEP_SECONDS = 2.0
USER_AGENT = "fairytales-dataset-bot/1.0 (contact: oleksandr.sakalosh0@gmail.com)"

WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)

START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)


@dataclass
class Book:
    ebook_id: int
    title: str
    author: Optional[str]
    book_page_url: str


def ensure_dirs():
    os.makedirs(TEXT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def clean_gutenberg_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n")
    m = START_RE.search(text)
    if m:
        text = text[m.end():]
    m2 = END_RE.search(text)
    if m2:
        text = text[:m2.start()]
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip()
    return text + "\n"


def count_words(text: str) -> int:
    # Proxy for word-level tokens (depends on your exact preprocessing/tokenizer)
    return len(WORD_RE.findall(text))


def http_get(session: requests.Session, url: str) -> str:
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.text


def parse_shelf_books(html: str, base_url: str) -> List[Book]:
    soup = BeautifulSoup(html, "lxml")
    out: List[Book] = []
    for a in soup.select("li.booklink > a.link[href*='/ebooks/']"):
        href = a.get("href")
        if not href:
            continue
        m = re.search(r"/ebooks/(\d+)", href)
        if not m:
            continue
        ebook_id = int(m.group(1))
        title_el = a.select_one(".title")
        author_el = a.select_one(".subtitle")
        title = title_el.get_text(strip=True) if title_el else f"ebook_{ebook_id}"
        author = author_el.get_text(strip=True) if author_el else None
        book_page_url = urljoin(base_url, href)
        out.append(Book(ebook_id=ebook_id, title=title, author=author, book_page_url=book_page_url))
    return out


def extract_bib_field(soup: BeautifulSoup, field_name: str) -> Optional[str]:
    """
    Gutenberg book pages have a bibliographic table-like layout.
    This finds a row where a header cell contains `field_name` (e.g., "Language")
    and returns the corresponding value cell text.
    """
    target = field_name.strip().lower()

    # Common pattern: <tr><th scope="row">Language</th><td>English</td></tr>
    for th in soup.find_all(["th", "td"]):
        txt = th.get_text(" ", strip=True).strip().lower()
        if txt == target:
            tr = th.find_parent("tr")
            if not tr:
                continue
            cells = tr.find_all(["td", "th"])
            if len(cells) >= 2:
                # value is usually the next cell
                val = cells[1].get_text(" ", strip=True)
                return val.strip() if val else None

    # Fallback: search for any element with text starting with 'Language'
    # and try to read the next sibling text.
    for el in soup.find_all(string=re.compile(r"^\s*Language\s*$", re.I)):
        parent = el.parent
        tr = parent.find_parent("tr") if parent else None
        if tr:
            tds = tr.find_all("td")
            if tds:
                return tds[0].get_text(" ", strip=True).strip()
    return None


def parse_book_page_txt_url_and_language(book_page_html: str, base_url: str) -> tuple[Optional[str], Optional[str]]:
    soup = BeautifulSoup(book_page_html, "lxml")

    # Language
    language = extract_bib_field(soup, "Language")

    # Plain text UTF-8 URL
    txt_url = None
    for a in soup.select("a[href]"):
        label = a.get_text(" ", strip=True).lower()
        if "plain text utf-8" in label:
            txt_url = urljoin(base_url, a["href"])
            break

    return txt_url, language


def fallback_txt_url(ebook_id: int) -> str:
    return f"https://www.gutenberg.org/ebooks/{ebook_id}.txt.utf-8"


def save_text(ebook_id: int, cleaned_text: str) -> str:
    path = os.path.join(TEXT_DIR, f"{ebook_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    return path


def load_existing_metadata() -> tuple[Dict[int, dict], int]:
    if not os.path.exists(META_CSV):
        return {}, 0

    rows: Dict[int, dict] = {}
    total = 0
    with open(META_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                eid = int(row["ebook_id"])
            except Exception:
                continue
            rows[eid] = row
            try:
                total += int(row.get("word_count", "0"))
            except Exception:
                pass
    return rows, total


def append_metadata_row(row: dict):
    file_exists = os.path.exists(META_CSV)
    fieldnames = [
        "ebook_id", "title", "author", "language",
        "book_page_url", "txt_url", "txt_path",
        "sha256", "byte_len", "char_len", "word_count",
        "downloaded_at_utc",
    ]
    with open(META_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def is_english(language_field: Optional[str]) -> bool:
    """
    Gutenberg may list multiple languages; keep it simple:
    accept if the string contains 'English' (case-insensitive).
    """
    if not language_field:
        return False
    return "english" in language_field.lower()


def build_dataset():
    ensure_dirs()
    existing, total_words = load_existing_metadata()
    print(f"[init] already have {len(existing)} books, total_words={total_words}")

    with requests.Session() as s:
        s.headers.update({"User-Agent": USER_AGENT})

        # Collect unique books from shelf pages
        books_by_id: Dict[int, Book] = {}
        for shelf_url in BOOKSHELF_PAGES:
            print(f"[shelf] {shelf_url}")
            html = http_get(s, shelf_url)
            for b in parse_shelf_books(html, shelf_url):
                books_by_id[b.ebook_id] = b
            time.sleep(REQUEST_SLEEP_SECONDS)

        books = [books_by_id[k] for k in sorted(books_by_id)]
        print(f"[found] {len(books)} unique books")

        newly_added = 0
        newly_words = 0
        skipped_non_en = 0
        skipped_no_lang = 0

        for b in books:
            if b.ebook_id in existing:
                continue

            print(f"[book] {b.ebook_id} — {b.title}")

            # Fetch book page: get language + txt url
            txt_url = None
            language = None
            try:
                book_html = http_get(s, b.book_page_url)
                txt_url, language = parse_book_page_txt_url_and_language(book_html, b.book_page_url)
            except Exception as e:
                print(f"  [warn] book page fetch/parse failed: {e}")

            if not language:
                skipped_no_lang += 1
                print("  [skip] no language field found")
                time.sleep(REQUEST_SLEEP_SECONDS)
                continue

            if not is_english(language):
                skipped_non_en += 1
                print(f"  [skip] language={language!r}")
                time.sleep(REQUEST_SLEEP_SECONDS)
                continue

            if not txt_url:
                txt_url = fallback_txt_url(b.ebook_id)

            # Download and save text
            try:
                raw_txt = http_get(s, txt_url)
            except Exception as e:
                print(f"  [warn] failed to download text: {e}")
                time.sleep(REQUEST_SLEEP_SECONDS)
                continue

            cleaned = clean_gutenberg_text(raw_txt)

            word_count = count_words(cleaned)
            char_len = len(cleaned)
            byte_len = len(cleaned.encode("utf-8"))
            digest = sha256_text(cleaned)

            txt_path = save_text(b.ebook_id, cleaned)

            row = {
                "ebook_id": b.ebook_id,
                "title": b.title,
                "author": b.author or "",
                "language": language,
                "book_page_url": b.book_page_url,
                "txt_url": txt_url,
                "txt_path": txt_path,
                "sha256": digest,
                "byte_len": byte_len,
                "char_len": char_len,
                "word_count": word_count,
                "downloaded_at_utc": utc_now_iso(),
            }
            append_metadata_row(row)

            newly_added += 1
            newly_words += word_count
            total_words += word_count

            print(f"  saved: {txt_path} | words={word_count} | total_words={total_words}")

            time.sleep(REQUEST_SLEEP_SECONDS)

        print("\n[done]")
        print(f"  new_books={newly_added}")
        print(f"  skipped_non_english={skipped_non_en}")
        print(f"  skipped_no_language_field={skipped_no_lang}")
        print(f"  new_words={newly_words}")
        print(f"  total_books={len(existing) + newly_added}")
        print(f"  total_words={total_words}")
        print(f"  metadata: {META_CSV}")
        print(f"  texts: {TEXT_DIR}/")


if __name__ == "__main__":
    build_dataset()