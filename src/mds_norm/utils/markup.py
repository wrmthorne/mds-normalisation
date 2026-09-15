from pathlib import Path

HTML_TAGS = (Path(__file__).parent / "html_tags.txt").read_text().split("\n")
# Tags that end a line of text
LINE_TAGS = ("br", "hr", "li", "dt", "dd", "tr", "td", "th")
# Tags that open or close a block of text; everything else is inline and leaves no trace
BLOCK_TAGS = (
    "address",
    "article",
    "aside",
    "blockquote",
    "caption",
    "center",
    "details",
    "div",
    "dl",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "summary",
    "table",
    "tbody",
    "tfoot",
    "thead",
    "ul",
)


def tag_regex(names: tuple[str, ...] | list[str]) -> str:
    """Opening or closing tags for one set of element names, attributes and all"""
    return rf"(?i)<\s*/?\s*(?:{'|'.join(names)})(?:\s[^>]*)?\s*/?\s*>"


# tags, comments and declarations
HTML_TAG_RE = rf"{tag_regex(HTML_TAGS)}|<!(?:--|\[|[a-zA-Z])"
HTML_LINE_TAG_RE = tag_regex(LINE_TAGS)
HTML_BLOCK_TAG_RE = tag_regex(BLOCK_TAGS)
HTML_ENTITY_RE = r"&(?:nbsp|amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);"
