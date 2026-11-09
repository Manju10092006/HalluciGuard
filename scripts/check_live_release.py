"""Check real HTTP verification and optional auth; print no credentials.

Example: python scripts/check_live_release.py https://example.run.app --auth
For Vercel: use https://example.vercel.app/api/backend as the origin.
"""
import argparse
import json
import secrets
import sys
import uuid

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('origin')
    parser.add_argument('--auth', action='store_true', help='Create a synthetic smoke-test account and check login/history')
    parser.add_argument('--correction', action='store_true', help='Also run a deliberately false supplied draft through correction')
    args = parser.parse_args()
    origin = args.origin.rstrip('/')

    def call(path, **kwargs):
        response = requests.request(kwargs.pop('method', 'GET'), origin + path, timeout=290, **kwargs)
        response.raise_for_status()
        return response.json()

    health = call('/health?deep=true')
    print('deep_health:', health.get('status'), flush=True)
    token = None
    if args.auth:
        email = f'release-smoke-{uuid.uuid4().hex}@example.com'
        password = secrets.token_urlsafe(32)
        call('/auth/register', method='POST', json={'email': email, 'password': password, 'name': 'Release smoke test'})
        login = call('/auth/login', method='POST', json={'email': email, 'password': password})
        token = login['access_token']
        headers = {'Authorization': f'Bearer {token}'}
        assert call('/auth/me', headers=headers)['user']['email'] == email
        denied = requests.post(origin + '/auth/login', json={'email': email, 'password': 'wrong-password'}, timeout=30)
        assert denied.status_code == 401, 'Incorrect password was accepted'
        print('auth: server JWT, persisted account and password rejection passed', flush=True)

    headers = {'Authorization': f'Bearer {token}'} if token else {}
    payloads = [{'user_query': 'Who founded Microsoft?', 'domain': 'general'}]
    if args.correction:
        payloads.append({'user_query': 'Who founded Microsoft?', 'llm_response': 'Microsoft was founded by Snehith.', 'domain': 'general'})
    for payload in payloads:
        result = call('/verify', method='POST', json=payload, headers=headers)
        claims = (result.get('verifier') or {}).get('claim_evidence') or []
        passed = result.get('terminal_status') == 'accepted' and result.get('verification_status') == 'verified_and_accepted' and bool(claims) and all(c['verdict'] == 'verified' for c in claims)
        print(json.dumps({'execution_id': result.get('execution_id'), 'terminal_status': result.get('terminal_status'), 'verification_status': result.get('verification_status'), 'claims': len(claims), 'sources': sum(len(c.get('evidence', [])) for c in claims), 'detector_inference': (result.get('detector') or {}).get('inference_executed'), 'latency_ms': result.get('total_latency_ms'), 'passed': passed}), flush=True)
        if not passed or not (result.get('detector') or {}).get('inference_executed'):
            return 1
        if payload.get('llm_response') and not (result.get('reverification_result') or {}).get('passed'):
            print('Correction did not pass independent re-verification.', flush=True)
            return 1
    if token:
        records = call('/api/history', headers=headers)['history']
        assert records, 'Authenticated verification was not persisted to history'
        print('history: authenticated verification persisted', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
