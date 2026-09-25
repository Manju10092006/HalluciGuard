# Contributing to HalluciGuard

HalluciGuard is a multi-package research engineering project. Changes must preserve evidence provenance, explicit failure states and canonical inter-agent contracts.

## Setup

```powershell
git clone https://github.com/Manju10092006/HalluciGuard.git
cd HalluciGuard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Frontend development:

```powershell
cd frontend-v2
npm install
npm run dev
```

Never commit `.env`, runtime databases, build output, caches, model downloads, private tunnel configuration or provider credentials.

## Branch and commit workflow

1. Create a focused branch from the current default branch.
2. Keep changes inside the smallest responsible package.
3. Update canonical contracts only when the cross-agent API truly changes.
4. Add focused regression tests before broad integration tests.
5. Run `git diff --check` and review staged files for secrets.
6. Use descriptive commits such as `fix(detector): ...`, `docs: ...`, or `test(verifier): ...`.
7. Push normally and open a pull request; never force-push shared history without explicit repository-owner coordination.

## Architecture rules

- The Base LLM cannot certify its own output.
- Claim Analyzer classifies checkability, not truth.
- Detector output is triage, not factual evidence.
- n8n is a retrieval broker, not the Verifier.
- Verifier owns claim/evidence verdicts.
- Judge owns release/correct/retry/reject/abstain policy.
- Corrector edits only Judge-authorized claims using bound evidence.
- ReVerifier independently checks corrected content.
- Memory persists only accepted verified facts.
- Missing models, evidence or credentials must remain explicit degraded/failure states.

## Tests

Choose suites proportional to the change:

```powershell
python -m pytest -q halluciguard_detector/tests
python -m pytest -q orchestration/tests
python -m pytest -q agents/verifier_agent/tests
python -m pytest -q agents/judge_agent/tests
python -m pytest -q agents/corrector_agent/tests
python -m pytest -q agents/memory_agent/tests
python -m pytest -q services/tests tests
```

Network/model tests must identify the credentials and services they require. Do not describe mocked execution as a live model test.

## Pull request checklist

- [ ] Scope and motivation are clear.
- [ ] Working imports and public contracts remain compatible or migration is documented.
- [ ] Evidence, scores and execution flags come from real runtime state.
- [ ] New failure paths fail closed.
- [ ] Tests cover the regression or feature.
- [ ] Documentation reflects actual behavior.
- [ ] No credentials, personal machine paths, caches, databases or generated logs are staged.
- [ ] `git diff --check` passes.

## Documentation truth

Do not invent model names, benchmark numbers, sources, provider behavior or production-readiness claims. Cite repository artifacts for measurements and label live demonstrations separately from automated tests.

## Security

Follow [SECURITY.md](SECURITY.md). Suspected credential exposure belongs in a private report, not a public issue.
