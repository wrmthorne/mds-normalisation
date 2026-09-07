from pathlib import Path

HTML_TAGS = (Path(__file__).parent / "html_tags.txt").read_text().split("\n")
_NAMES = "|".join(HTML_TAGS)

# tags, comments and declarations
HTML_TAG_RE = rf"(?i)<\s*/?\s*(?:{_NAMES})(?:\s[^>]*)?\s*/?\s*>|<!(?:--|\[|[a-zA-Z])"
HTML_ENTITY_RE = r"&(?:nbsp|amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);"
