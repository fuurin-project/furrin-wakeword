from __future__ import annotations

import re


def normalize_phrase(value: str) -> str:
    phrase = " ".join(value.strip().split())
    if len(phrase.split()) < 2:
        raise ValueError("wake phrase must contain at least two words")
    return phrase


def phrase_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    if not slug:
        raise ValueError("wake phrase must contain ASCII letters or digits")
    return slug


def default_target_texts(value: str) -> tuple[str, ...]:
    phrase = normalize_phrase(value)
    return (
        phrase,
        phrase.lower(),
        f"{phrase}!",
        f"{phrase}?",
        phrase.replace(" ", ", ", 1),
        phrase.replace(" ", "... ", 1),
    )


def default_hard_negative_texts(value: str) -> tuple[str, ...]:
    phrase = normalize_phrase(value)
    words = phrase.split()
    prefix = words[0]
    keyword = " ".join(words[1:])
    candidates = [
        prefix,
        keyword,
        f"Okay {keyword}",
        f"Hello {keyword}",
        f"Hi {keyword}",
        f"{prefix} computer",
        f"{prefix} robot",
        "Play some music",
        "Stop the music",
        "Turn on the light",
        "What time is it",
        "Set a timer",
        "Open the door",
        "Good morning",
        "Volume up",
        "Volume down",
        "Thank you",
        "Please stop",
    ]
    return tuple(dict.fromkeys(candidates))


def negative_kind(text: str, value: str) -> str:
    normalized_text = " ".join(text.lower().strip(" ,.!?").split())
    words = normalize_phrase(value).lower().split()
    components = [words[0], " ".join(words[1:])]
    for index, component in enumerate(components):
        if normalized_text == component:
            return f"partial_{index}"
    return "hard_negative"


def is_partial_kind(kind: str) -> bool:
    return kind.startswith("partial_") or kind in {"hey_only", "pico_only"}
