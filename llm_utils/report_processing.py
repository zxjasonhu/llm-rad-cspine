import json
import re

def clean_json(json_str: str) -> str:
    """Robust cleaning of invalid JSON with multiple fallbacks"""
    json_str = re.sub(
        r'(//.*?$|/\*.*?\*/|\\\\[^\'"]*?$)',
        "",
        json_str,
        flags=re.MULTILINE | re.DOTALL,
    )

    json_str = (
        json_str.replace("False", "false")
        .replace("FALSE", "false")
        .replace("True", "true")
        .replace("TRUE", "true")
    )

    json_str = (
        json_str.replace(r"\/", "/")  # Unescape forward slashes
        .replace(r"\\\\", r"\\")  # Fix double-escaped backslashes
        .replace(r'\\"', r"\"")  # Fix escaped quotes
    )

    attempts = [
        lambda s: s,
        lambda s: s + "}" * (s.count("{") - s.count("}")),
        lambda s: s + "]" * (s.count("[") - s.count("]")),
        lambda s: re.sub(r",\s*([}\]])", r"\1", s),
    ]

    for attempt in attempts:
        try:
            repaired = attempt(json_str)
            json.loads(repaired)
            return repaired
        except (json.JSONDecodeError, TypeError):
            continue

    return json_str  # Final fallback

def extract_json_labels(text: str) -> dict:
    """Enhanced extraction with structural repair"""
    text = text.strip()

    text = text.replace('""', '"').replace('""null""', "null")

    for pattern in [
        r"```json(.*?)```",  # Markdown with json
        r"<json>(.*?)</json>",  # XML-style
        r"```(.*?)```",  # Any code block
        r"({.*})",  # Loose JSON match
    ]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            cleaned = clean_json(json_str)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    try:
        return json.loads(clean_json(text))
    except json.JSONDecodeError:
        return {}

def white_space_fix(text):
    _text = text.strip()
    _text = re.sub(r"\s+", " ", _text)
    return _text

def preprocess_report(report_text):
    report_text = report_text.replace("_x000D_", " ").replace("*", "")

    report_text = f"""Here is the CT radiology report you need to analyze:\n```plaintext\n{report_text}\n```"""
    report_text = white_space_fix(report_text)
    return report_text

