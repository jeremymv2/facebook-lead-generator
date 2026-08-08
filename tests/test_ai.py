import json
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from lead_agent.ai import (
    AIConfigurationError,
    AIProviderDisabledError,
    AIProviderError,
    AIResponseError,
    ClassificationContext,
    DraftResponse,
    GeminiAIProvider,
    GoogleGenAITransport,
    HeuristicAIProvider,
    LeadClassification,
    build_ai_provider,
)
from lead_agent.config import DEFAULT_SERVICES, Settings
from lead_agent.models import FacebookPost, LeadIntent

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "facebook_posts.json"


class FakeTransport:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []

    def generate_json(
        self,
        *,
        system_instruction: str,
        prompt: str,
        schema: dict[str, object],
    ) -> str:
        self.calls.append(
            {
                "system_instruction": system_instruction,
                "prompt": prompt,
                "schema": schema,
            }
        )
        return self.outputs.pop(0)


def context() -> ClassificationContext:
    return ClassificationContext(
        service_area="Louisville, Kentucky",
        service_radius_miles=50,
        enabled_services=tuple(DEFAULT_SERVICES),
        lead_threshold=75,
        max_input_characters=5000,
    )


def post(text: str, *, author_name: str | None = "Sarah Example") -> FacebookPost:
    return FacebookPost(
        external_post_id="fixture-123",
        group_id="fixture-group",
        group_name="Synthetic Fixture Group",
        author_name=author_name,
        post_text=text,
    )


def valid_classification_json(**overrides: object) -> str:
    values: dict[str, object] = {
        "service_category": "decks",
        "location": "Louisville",
        "intent": "hiring",
        "is_residential": True,
        "is_spam": False,
        "relevance_score": 95,
        "geographic_score": 100,
        "urgency_score": 85,
        "overall_score": 94,
        "confidence": 0.96,
        "reasoning_summary": "Strong local residential deck hiring request.",
    }
    values.update(overrides)
    return json.dumps(values)


def test_heuristic_provider_matches_classification_fixtures() -> None:
    fixtures = cast(list[dict[str, object]], json.loads(FIXTURE_PATH.read_text(encoding="utf-8")))
    provider = HeuristicAIProvider()

    for fixture in fixtures:
        result = provider.classify_post(post(cast(str, fixture["text"])), context())
        expected = cast(dict[str, object], fixture["expected"])

        if "service" in expected:
            assert result.service_category == expected["service"], fixture["name"]
        if "intent" in expected:
            assert result.intent.value == expected["intent"], fixture["name"]
        if "residential" in expected:
            assert result.is_residential is expected["residential"], fixture["name"]
        if "spam" in expected:
            assert result.is_spam is expected["spam"], fixture["name"]
        if "minimum_score" in expected:
            assert result.overall_score >= cast(int, expected["minimum_score"]), fixture["name"]
        if "maximum_score" in expected:
            assert result.overall_score <= cast(int, expected["maximum_score"]), fixture["name"]
        if "maximum_geographic_score" in expected:
            assert result.geographic_score <= cast(int, expected["maximum_geographic_score"]), (
                fixture["name"]
            )


def test_heuristic_draft_is_direct_and_locally_validated() -> None:
    provider = HeuristicAIProvider()
    source = post("Looking for someone in Louisville to repair our deck this week.")
    classification = provider.classify_post(source, context())

    draft = provider.draft_response(source, classification, context())

    assert not draft.response.startswith(("Hi ", "Hello ", "Hey "))
    assert "JJ Miller & Co." in draft.response
    assert "deck work" in draft.response
    assert "decks" not in draft.response
    assert "https://jjmillerco.com" in draft.response
    assert "Text me at 502-528-0858" in draft.response
    assert "free estimate" in draft.response.casefold()
    assert "message" not in draft.response.casefold()
    assert len(draft.response) <= 300


def test_heuristic_lawn_draft_uses_natural_service_language() -> None:
    provider = HeuristicAIProvider()
    source = post("Need someone to come quote our yard for mowing in Louisville.")
    classification = provider.classify_post(source, context())

    draft = provider.draft_response(source, classification, context())

    assert classification.service_category == "landscaping"
    assert "free estimates for lawn services" in draft.response.casefold()
    assert "we'd be" in draft.response.casefold()
    assert "landscaping" not in draft.response.casefold()
    assert "need help" not in draft.response.casefold()


def test_heuristic_matches_plural_structural_repairs_in_crawl_space() -> None:
    provider = HeuristicAIProvider()

    classification = provider.classify_post(
        post("Need estimates for structural repairs in crawl space"),
        context(),
    )

    assert classification.service_category == "structural_repairs"
    assert classification.intent is LeadIntent.HIRING
    assert classification.overall_score >= context().lead_threshold


def test_heuristic_recognizes_who_are_you_using_as_recommendation() -> None:
    provider = HeuristicAIProvider()

    classification = provider.classify_post(
        post("INVESTORS: Who are you using for interior paint jobs and flooring installation?"),
        context(),
    )

    assert classification.service_category == "flooring"
    assert classification.intent is LeadIntent.RECOMMENDATION
    assert classification.overall_score >= context().lead_threshold


def test_heuristic_rejects_commercial_property_request() -> None:
    provider = HeuristicAIProvider()

    classification = provider.classify_post(
        post("Need someone for a landscape permit on a commercial property."),
        context(),
    )

    assert classification.is_residential is False
    assert classification.overall_score <= 40


@pytest.mark.parametrize(
    "text",
    [
        (
            "Turn your backyard into your favorite place. We'll build a deck made to last and "
            "designed for your home."
        ),
        (
            "I can schedule your estimate for this week and start soon. We are the professional "
            "choice in landscaping."
        ),
        (
            "Whether it's a small repair or complete renovation, we'd love the opportunity to "
            "earn your business. Our services include painting, decks, and structural repairs."
        ),
        "Floors N More offers tile installation. #FreeEstimate #LicensedAndInsured",
    ],
)
def test_heuristic_suppresses_realistic_contractor_advertisements(text: str) -> None:
    classification = HeuristicAIProvider().classify_post(post(text), context())

    assert classification.intent is LeadIntent.COMPETITOR_ADVERTISEMENT
    assert classification.overall_score <= 10


def test_heuristic_recognizes_desperate_mowing_company_request() -> None:
    classification = HeuristicAIProvider().classify_post(
        post(
            "NEED: I am in desperate need of a mowing company for three downtown properties. "
            "The grass is very high and inspectors are issuing citations. Please send referrals."
        ),
        context(),
    )

    assert classification.service_category == "landscaping"
    assert classification.intent is LeadIntent.RECOMMENDATION
    assert classification.overall_score >= context().lead_threshold


def test_heuristic_uses_repeated_project_terms_for_primary_service() -> None:
    classification = HeuristicAIProvider().classify_post(
        post(
            "Looking for a reliable deck contractor. Replace 30 deck boards, repair the deck "
            "supports, sand the deck, remove peeling paint, and apply two coats of deck coating. "
            "Do not paint over failing paint. I need an itemized estimate."
        ),
        context(),
    )

    assert classification.service_category == "decks"
    assert classification.intent is LeadIntent.HIRING
    assert classification.overall_score >= context().lead_threshold


@pytest.mark.parametrize(
    ("text", "expected_service"),
    [
        ("Need someone for brick masonry repair in Louisville.", "masonry"),
        ("Looking for help with a leaking roof repair this week.", "roof_repair"),
        ("Need soffit and siding repair in Louisville.", "exterior_repairs"),
        ("Looking for gutter cleaning and downspout repair.", "gutters_and_drainage"),
        ("Need a pergola constructed in our backyard.", "outdoor_structures"),
        ("Looking for selective demolition and wall removal.", "demolition"),
        (
            "Need someone for TV mounting and curtain rod installation.",
            "installations_and_mounting",
        ),
        (
            "Looking for attic insulation and air-sealing improvements.",
            "insulation_and_air_sealing",
        ),
        ("Need a minor leak repair and shutoff-valve replacement.", "minor_plumbing_repairs"),
        ("Looking for dishwasher installation this week.", "appliance_installation"),
        ("Need a ceiling fan installation in Louisville.", "electrical_fixtures"),
        ("Looking for dryer vent repair and cleaning.", "ventilation"),
        ("Need rental turnover and property maintenance help.", "property_maintenance"),
        ("Looking for construction debris removal and hauling.", "cleanup_and_hauling"),
        ("Need remodeling consultation and project coordination.", "project_coordination"),
    ],
)
def test_heuristic_matches_expanded_published_services(
    text: str,
    expected_service: str,
) -> None:
    classification = HeuristicAIProvider().classify_post(post(text), context())

    assert expected_service in DEFAULT_SERVICES
    assert classification.service_category == expected_service
    assert classification.overall_score >= context().lead_threshold


def test_roof_repair_allowlist_does_not_enable_unspecified_roof_replacement() -> None:
    classification = HeuristicAIProvider().classify_post(
        post("Looking for a contractor to replace an entire roof in Louisville."),
        context(),
    )

    assert classification.service_category is None
    assert classification.overall_score <= 40


def test_heuristic_suppresses_open_house_real_estate_advertisement() -> None:
    classification = HeuristicAIProvider().classify_post(
        post(
            "OPEN HOUSE THIS SUNDAY! This beautifully updated home has new flooring, a new roof, "
            "and a $15,000 price improvement."
        ),
        context(),
    )

    assert classification.intent is LeadIntent.SELLING
    assert classification.overall_score <= 10


def test_heuristic_caps_contractor_subcontracting_request_as_non_residential() -> None:
    classification = HeuristicAIProvider().classify_post(
        post(
            "Breed Renovations: I'm looking for someone who does roofing. I don't want to sub "
            "the job out completely because I want to work and learn."
        ),
        context(),
    )

    assert classification.is_residential is False
    assert classification.overall_score <= 40


def test_heuristic_suppresses_provider_scheduling_advertisement() -> None:
    classification = HeuristicAIProvider().classify_post(
        post(
            "Finished this natural stone retaining wall for a customer. Contact us to schedule "
            "your free estimate."
        ),
        context(),
    )

    assert classification.intent is LeadIntent.COMPETITOR_ADVERTISEMENT
    assert classification.overall_score <= 10


def test_heuristic_respects_private_contact_only_instruction() -> None:
    classification = HeuristicAIProvider().classify_post(
        post(
            "Looking for a reliable deck contractor. I don't respond to comments. Seriously DM "
            "inquiries only. Replace 30 deck boards and repair the supports."
        ),
        context(),
    )

    assert classification.service_category == "decks"
    assert classification.intent is LeadIntent.PRIVATE_CONTACT_ONLY
    assert classification.overall_score <= 10


def test_heuristic_draft_does_not_use_an_unsafe_author_fragment() -> None:
    provider = HeuristicAIProvider()
    source = post(
        "Looking for someone in Louisville to repair our deck this week.",
        author_name="<script>",
    )
    classification = provider.classify_post(source, context())

    draft = provider.draft_response(source, classification, context())

    assert "Hi there" not in draft.response
    assert "<script>" not in draft.response


def test_heuristic_drafts_vary_by_stable_post_content() -> None:
    provider = HeuristicAIProvider()
    drafts: set[str] = set()
    for index in range(10):
        source = post(f"Looking for deck repair in Louisville this week. Fixture {index}.")
        classification = provider.classify_post(source, context())
        drafts.add(provider.draft_response(source, classification, context()).response)

    assert len(drafts) > 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"intent": "competitor_advertisement", "overall_score": 80},
        {"intent": "private_contact_only", "overall_score": 80},
        {"intent": "resolved", "overall_score": 80},
        {"intent": "advice", "overall_score": 60},
        {"is_residential": False, "overall_score": 70},
        {"service_category": None, "relevance_score": 90},
    ],
)
def test_classification_rejects_semantically_unsafe_scores(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        LeadClassification.model_validate_json(valid_classification_json(**overrides))


def test_draft_rejects_missing_identity_and_promotional_spam() -> None:
    with pytest.raises(ValidationError, match="identify"):
        DraftResponse(
            response=("Free estimates. Text me at 502-528-0858 or visit https://jjmillerco.com.")
        )
    with pytest.raises(ValidationError, match="promotional"):
        DraftResponse(
            response=(
                "JJ Miller & Co. — WE DO EVERYTHING CALL US NOW. Free estimates. Text me at "
                "502-528-0858 or visit https://jjmillerco.com."
            )
        )


@pytest.mark.parametrize(
    ("response", "error"),
    [
        (
            "JJ Miller & Co. offers free estimates. Text me at 502-528-0858 or visit "
            "jjmillerco.com.",
            "fully qualified",
        ),
        (
            "JJ Miller & Co. offers free estimates. Message me or visit "
            "https://jjmillerco.com for details.",
            "company phone number",
        ),
        (
            "JJ Miller & Co. can help. Text me at 502-528-0858 or visit https://jjmillerco.com.",
            "estimates are free",
        ),
        (
            "Hi there, JJ Miller & Co. offers free estimates. Text me at 502-528-0858 or visit "
            "https://jjmillerco.com.",
            "generic greeting",
        ),
        (
            "JJ Miller & Co. says thanks and offers free estimates. Text me at 502-528-0858. "
            "https://jjmillerco.com",
            "generic greeting",
        ),
        (
            "JJ Miller & Co. offers free estimates. Text me at 502-528-0858 or send me a message. "
            "https://jjmillerco.com",
            "Facebook messaging",
        ),
        (
            "Need help with lawn services? JJ Miller & Co. offers free estimates. Text me at "
            "502-528-0858. https://jjmillerco.com",
            "rhetorical",
        ),
    ],
)
def test_draft_enforces_business_contact_and_voice_rules(response: str, error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        DraftResponse(response=response)


def test_gemini_provider_uses_structured_schemas_and_minimal_post_metadata() -> None:
    transport = FakeTransport(
        [
            valid_classification_json(),
            json.dumps(
                {
                    "response": (
                        "JJ Miller & Co. can help with your deck project. Free estimates. Text me "
                        "at 502-528-0858 or visit https://jjmillerco.com."
                    )
                }
            ),
        ]
    )
    provider = GeminiAIProvider(model="gemini-fixture", transport=transport)
    source = post("Looking for someone in Louisville to repair our deck this week.")

    classification = provider.classify_post(source, context())
    draft = provider.draft_response(source, classification, context())

    assert classification.intent is LeadIntent.HIRING
    assert "JJ Miller & Co." in draft.response
    assert len(transport.calls) == 2
    assert "post_text" in cast(str, transport.calls[0]["prompt"])
    assert "lawn services" in cast(str, transport.calls[1]["prompt"])
    assert "rhetorical question" in cast(str, transport.calls[1]["prompt"])
    assert "fixture-group" not in cast(str, transport.calls[0]["prompt"])
    assert "properties" in cast(dict[str, object], transport.calls[0]["schema"])


def test_gemini_provider_rejects_invalid_json_and_unapproved_services() -> None:
    invalid = GeminiAIProvider(model="gemini-fixture", transport=FakeTransport(["not json"]))
    with pytest.raises(AIResponseError, match="validation"):
        invalid.classify_post(post("Need deck repair in Louisville."), context())

    unapproved = GeminiAIProvider(
        model="gemini-fixture",
        transport=FakeTransport([valid_classification_json(service_category="roofing")]),
    )
    with pytest.raises(AIResponseError, match="allowlist"):
        unapproved.classify_post(post("Need a roofer in Louisville."), context())


def test_google_transport_wraps_sdk_errors_without_leaking_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingInteractions:
        def create(self, **kwargs: object) -> object:
            del kwargs
            raise RuntimeError("raw provider response and request content")

    class FakeClient:
        interactions = FailingInteractions()

    monkeypatch.setattr("google.genai.Client", lambda **kwargs: FakeClient())
    transport = GoogleGenAITransport(api_key="placeholder", model="fixture", timeout_seconds=5)

    with pytest.raises(AIProviderError) as captured:
        transport.generate_json(system_instruction="safe", prompt="data", schema={})

    assert "raw provider" not in str(captured.value)


def test_google_transport_returns_sdk_output_text(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    class SuccessfulInteractions:
        def create(self, **kwargs: object) -> object:
            calls.append(kwargs)

            class Response:
                output_text = '{"result":"ok"}'

            return Response()

    class FakeClient:
        interactions = SuccessfulInteractions()

    def fake_client(**kwargs: object) -> FakeClient:
        del kwargs
        return FakeClient()

    monkeypatch.setattr("google.genai.Client", fake_client)
    transport = GoogleGenAITransport(api_key="placeholder", model="fixture", timeout_seconds=5)

    output = transport.generate_json(
        system_instruction="safe",
        prompt="data",
        schema={"type": "object"},
    )

    assert output == '{"result":"ok"}'
    assert calls[0]["model"] == "fixture"
    assert calls[0]["system_instruction"] == "safe"
    assert cast(dict[str, object], calls[0]["response_format"])["mime_type"] == ("application/json")


def test_provider_factory_defaults_disabled_and_requires_gemini_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    profile = tmp_path.parent / "browser-profile"

    with pytest.raises(AIProviderDisabledError):
        build_ai_provider(Settings(_env_file=None, facebook_profile_path=profile))
    assert isinstance(
        build_ai_provider(
            Settings(
                _env_file=None,
                facebook_profile_path=profile,
                ai_provider="heuristic",
            )
        ),
        HeuristicAIProvider,
    )
    with pytest.raises(AIConfigurationError, match="GEMINI_API_KEY"):
        build_ai_provider(
            Settings(
                _env_file=None,
                facebook_profile_path=profile,
                ai_provider="gemini",
            )
        )
