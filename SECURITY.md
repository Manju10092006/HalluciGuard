# Security policy

## Credentials

Never commit API keys, tokens, passwords, private keys, webhook secrets, authenticated URLs, cookies, runtime databases or `.env` files. Use `.env.example` for names and placeholders only; store real values in the ignored `.env` or a deployment secret manager.

Sensitive backend variables include Groq, Gemini, OpenRouter, Tavily, n8n, JWT, Google identity, domain API and Memory API credentials. No server credential should use a `NEXT_PUBLIC_` prefix.

If a credential is exposed:

1. Revoke or rotate it immediately.
2. Replace the tracked value with an environment reference or placeholder.
3. Review Git history and provider access logs.
4. Notify maintainers privately; never repost the credential in an issue.

Repository cleanup found a historical Tavily-shaped value in documentation and replaced the current occurrence with a placeholder. Rotation is still required because changing the latest commit does not erase Git history.

## Reporting

Use a private GitHub security advisory when available. If private reporting is unavailable, contact the repository owner without publishing credentials or exploit details.

Include the affected component, impact, reproduction conditions, proposed mitigation and whether credentials may have been exposed.

## Operational guidance

- Require a strong `JWT_SECRET` in production.
- Restrict CORS to deployed client origins.
- Authenticate n8n webhooks and rotate tunnel URLs/secrets.
- Keep runtime SQLite, vector and graph data outside Git.
- Treat retrieved web content as untrusted input.
- Preserve fail-closed behavior when models, providers or evidence are unavailable.
- Require human review for high-stakes medical, legal, financial, security and safety use.
