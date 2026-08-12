from lazyportfolio.advisor.redaction import redact_pii


def test_email_is_redacted() -> None:
    assert redact_pii("contact me at analyst.example@example.com please") == (
        "contact me at [redacted-email] please"
    )


def test_phone_number_is_redacted() -> None:
    assert redact_pii("call 555-123-4567 tomorrow") == "call [redacted-phone] tomorrow"
    assert redact_pii("call +1 555-123-4567 tomorrow") == "call [redacted-phone] tomorrow"


def test_long_digit_run_shaped_like_an_account_or_ssn_is_redacted() -> None:
    assert redact_pii("account 123456789012 is overdrawn") == (
        "account [redacted-number] is overdrawn"
    )


def test_short_numbers_are_left_alone() -> None:
    """A confidence value, a year, a node id suffix -- ordinary short
    numbers in Node Advisor text must not be mangled by the PII redactor."""

    text = "confidence 0.6, year 2026, node equity-3"
    assert redact_pii(text) == text


def test_text_with_no_pii_is_returned_unchanged() -> None:
    text = "SPY should outperform TLT given the current regime."
    assert redact_pii(text) == text


def test_multiple_pii_instances_in_one_string_are_all_redacted() -> None:
    text = "email first@example.com or second@example.org, or call 555-000-1111"
    result = redact_pii(text)
    assert "first@example.com" not in result
    assert "second@example.org" not in result
    assert "555-000-1111" not in result
    assert result.count("[redacted-email]") == 2
    assert result.count("[redacted-phone]") == 1


def test_email_local_part_digits_are_not_double_redacted_as_a_long_number() -> None:
    """Redacting emails first means the digit run inside an email's local
    part never independently matches the long-digit-run pattern afterward."""

    result = redact_pii("contact 123456789012@example.com now")
    assert result == "contact [redacted-email] now"
