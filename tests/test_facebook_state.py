import pytest

from lead_agent.facebook_state import FacebookPageState, assess_facebook_page


@pytest.mark.parametrize(
    ("url", "text", "expected"),
    [
        (
            "https://www.facebook.com/groups/123",
            "Louisville Homeowners Recent activity",
            FacebookPageState.NORMAL,
        ),
        (
            "https://www.facebook.com/login/",
            "Log into Facebook",
            FacebookPageState.LOGIN_REQUIRED,
        ),
        (
            "https://www.facebook.com/checkpoint/123",
            "Confirm your identity",
            FacebookPageState.CHECKPOINT,
        ),
        (
            "https://www.facebook.com/captcha/",
            "Complete the security check",
            FacebookPageState.CAPTCHA,
        ),
        (
            "https://example.com/redirect",
            "Facebook",
            FacebookPageState.UNEXPECTED,
        ),
    ],
)
def test_page_assessment_fails_closed(
    url: str,
    text: str,
    expected: FacebookPageState,
) -> None:
    assert assess_facebook_page(url, text).state is expected


def test_captcha_takes_precedence_over_checkpoint_wording() -> None:
    assessment = assess_facebook_page(
        "https://www.facebook.com/checkpoint/captcha/",
        "Confirm your identity. Enter the text you see.",
    )

    assert assessment.state is FacebookPageState.CAPTCHA
