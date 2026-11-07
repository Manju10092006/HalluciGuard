import pytest
from services.character_regenerator import CharacterRegenerator
from services.tests.test_character_regenerator import _request, _FakeService, _Gen
from orchestration.schemas import ClaimReport

@pytest.mark.asyncio
async def test_preserved_clause_in_repaired_sentence_cannot_be_silently_removed():
    request=_request()
    original='Java was created by Dennis Ritchie in 1972 and Java runs on a virtual machine.'
    request=request.model_copy(update={'original_response':original,
        'claims_to_correct':[request.claims_to_correct[0].model_copy(update={'claim_text':original})],
        'claims_to_preserve':[ClaimReport(claim_id='keep',claim_text='Java runs on a virtual machine',verdict='verified')]})
    result=await CharacterRegenerator(service=_FakeService([_Gen('Java was created by James Gosling.')])).regenerate(request)
    assert result.status=='failed'
    assert result.failure_category=='PRESERVATION_VIOLATION'
    assert result.corrected_text=='' and result.original_text==original
