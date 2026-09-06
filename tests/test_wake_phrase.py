import pytest

from oww_distill.wake_phrase import (
    default_hard_negative_texts,
    default_target_texts,
    is_partial_kind,
    negative_kind,
    normalize_phrase,
    phrase_slug,
)


def test_generic_phrase_defaults_include_both_components() -> None:
    phrase = normalize_phrase("  Hey   Nordic ")
    assert phrase == "Hey Nordic"
    assert default_target_texts(phrase)[0] == phrase
    assert default_hard_negative_texts(phrase)[:2] == ("Hey", "Nordic")
    assert negative_kind("Hey!", phrase) == "partial_0"
    assert negative_kind("Nordic", phrase) == "partial_1"
    assert negative_kind("Hey robot", phrase) == "hard_negative"


def test_phrase_slug_accepts_explicit_single_token_name() -> None:
    assert phrase_slug("Nordic-v2") == "nordic_v2"


def test_phrase_requires_two_words() -> None:
    with pytest.raises(ValueError, match="at least two words"):
        normalize_phrase("Nordic")


def test_partial_kind_keeps_legacy_hey_pico_compatibility() -> None:
    assert is_partial_kind("partial_0")
    assert is_partial_kind("pico_only")
    assert not is_partial_kind("hard_negative")
