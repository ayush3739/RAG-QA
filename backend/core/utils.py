# backend/core/utils.py
import re
def simple_tokenize(text: str):
    return re.findall(r"\w+", text.lower())