from __future__ import annotations

import pytest

from adapters.general import GeneralAdapter
from adapters.healthcare import HealthcareAdapter
from adapters.legal_general import LegalGeneralAdapter
from config.settings import get_settings
from models.model_manager import get_model_manager


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.text = ""

    def json(self) -> dict:
        return self._payload


class FakeClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    async def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return FakeResponse(self.responses.pop(0))


def test_model_manager_uses_official_huggingface_ids() -> None:
    settings = get_settings()
    status = get_model_manager().status()

    assert settings.zero_shot_model == "facebook/bart-large-mnli"
    # NLI_MODEL may be overridden to a local path via .env
    assert settings.nli_model in (
        "cross-encoder/nli-deberta-v3-base",
        r"C:\temp\test_nli",
    ), f"Unexpected nli_model: {settings.nli_model}"
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.reranker_model == "BAAI/bge-reranker-large"
    assert settings.embedding_model in status


@pytest.mark.anyio
async def test_healthcare_pubmed_central_uses_ncbi_eutilities() -> None:
    client = FakeClient([
        {"esearchresult": {"idlist": ["123456"]}},
        {"result": {"123456": {"title": "PMC Evidence", "pubdate": "2024 Jan"}}},
    ])
    adapter = HealthcareAdapter()

    passages = await adapter._search_pubmed_central(client, "metformin", 1)  # type: ignore[arg-type]

    assert client.calls[0]["url"] == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    assert client.calls[0]["params"]["db"] == "pmc"  # type: ignore[index]
    assert client.calls[1]["url"] == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    assert passages[0].source == "pubmed_central"
    assert passages[0].url == "https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/"


@pytest.mark.anyio
async def test_legal_adapter_uses_courtlistener_not_curated_facts() -> None:
    client = FakeClient([
        {
            "results": [
                {
                    "caseName": "Miranda v. Arizona",
                    "absolute_url": "/opinion/123/miranda-v-arizona/",
                    "snippet": "A landmark self-incrimination decision.",
                    "dateFiled": "1966-06-13",
                    "cluster_id": 123,
                }
            ]
        }
    ])
    adapter = LegalGeneralAdapter()

    passages = await adapter._search_courtlistener(client, "Miranda v Arizona", 1)  # type: ignore[arg-type]

    assert client.calls[0]["url"] == "https://www.courtlistener.com/api/rest/v4/search/"
    assert client.calls[0]["params"]["type"] == "o"  # type: ignore[index]
    assert passages[0].source == "courtlistener"
    assert passages[0].source_id == "courtlistener_123"


@pytest.mark.anyio
async def test_general_adapter_uses_wikipedia_rest_search(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient([
        {"pages": [{"title": "Jupiter", "key": "Jupiter", "excerpt": "Largest planet."}]}
    ])
    monkeypatch.setattr("adapters.general.get_client", lambda: client)

    passages = await GeneralAdapter().search("Jupiter", k=1)

    assert client.calls[0]["url"] == "https://en.wikipedia.org/w/rest.php/v1/search/page"
    assert client.calls[0]["params"] == {"q": "Jupiter", "limit": 1}
    assert passages[0].source == "wikipedia"
