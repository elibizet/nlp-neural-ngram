import re

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)