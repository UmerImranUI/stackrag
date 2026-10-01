
from bs4 import BeautifulSoup
import html
import re


def clean_html(raw_html: str) -> str:
    """
    Convert Stack Exchange HTML into clean readable text.

    - Removes HTML markup
    - Decodes HTML entities
    - Preserves code blocks
    - Prevents text from different HTML elements being joined
    - Normalizes whitespace
    """
    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")

    # Preserve code blocks before removing HTML markup.
    for code in soup.find_all("code"):
        code_text = code.get_text("\n", strip=False)
        code.replace_with(
            f"\nCODE:\n{code_text}\nEND_CODE\n"
        )

    # Add spacing around block-level elements so adjacent
    # text nodes don't get accidentally joined.
    block_tags = [
        "p",
        "div",
        "li",
        "ul",
        "ol",
        "blockquote",
        "pre",
        "table",
        "tr",
        "th",
        "td",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    ]

    for tag in soup.find_all(block_tags):
        tag.insert_before("\n")
        tag.insert_after("\n")

    text = soup.get_text(" ")

    # Decode HTML entities such as:
    # &#x20; -> space
    # &amp;  -> &
    text = html.unescape(text)

    # Remove non-breaking spaces.
    text = text.replace("\xa0", " ")

    # Normalize spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize line endings.
    text = re.sub(r"\r\n?", "\n", text)

    # Remove spaces around newlines.
    text = re.sub(r" *\n *", "\n", text)

    # Collapse excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def build_document_text(
    title: str,
    question_body: str,
    answer_body: str,
) -> str:
    """
    Build the semantic text that will later be chunked and embedded.

    Each RAG document contains:
    - Question title
    - Question
    - Answer
    """
    question = clean_html(question_body)
    answer = clean_html(answer_body)

    sections = [
        f"Title:\n{title.strip()}",
        f"Question:\n{question}",
        f"Answer:\n{answer}",
    ]

    return "\n\n".join(
        section
        for section in sections
        if section.strip()
    )
