import re


def normalize_answer(text: str) -> str:
    """Normalize short free-form answers for exact-match evaluation."""
    normalized = (text or "").strip().lower().replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"[\,\.\?\!\:\;\"\'\(\)\[\]\{\}]", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    yes_set = {"yes", "y", "yeah", "yep", "true", "correct"}
    no_set = {"no", "n", "nope", "false", "incorrect"}
    if normalized in yes_set:
        return "yes"
    if normalized in no_set:
        return "no"
    return normalized


def extract_answer_text(text: str) -> str:
    """Strip the common answer prefix from generated text when present."""
    if "Answer:" in text:
        return text.split("Answer:", 1)[-1].strip()
    return text.strip()
