"""Print a private candidate deployment plan; this program never executes it.

Only supply existing secret VERSION resource names, never secret values.
The parent task must review financial scope and persistence limitations first.
"""
from __future__ import annotations

import argparse
import json
import re

PROJECT = "project-836457aa-3a05-4a11-8f8"
REGION = "asia-south1"
SERVICE = "halluciguard-staging-20261024"
IMAGE = ("asia-south1-docker.pkg.dev/" + PROJECT +
         "/cloud-run-source-deploy/halluciguard/halluciguard-detector@sha256:")


def plan(digest: str, account: str, openrouter_secret: str, jwt_secret: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Supply the successfully built immutable image SHA256")
    if not re.fullmatch(r"[A-Za-z0-9._-]+@" + re.escape(PROJECT) + r"\.iam\.gserviceaccount\.com", account):
        raise ValueError("Use an existing runtime service account in this project")
    for secret in (openrouter_secret, jwt_secret):
        if not re.fullmatch(r"[A-Za-z0-9_-]+:[0-9]+", secret):
            raise ValueError("Use existing secret-name:numeric-version references, not values or latest")
    command = [
        "gcloud", "run", "deploy", SERVICE, "--project=" + PROJECT, "--region=" + REGION,
        "--image=" + IMAGE + digest, "--service-account=" + account,
        "--cpu=2", "--memory=8Gi", "--concurrency=1", "--min-instances=0", "--max-instances=1",
        "--timeout=300s", "--cpu-throttling", "--no-cpu-boost", "--execution-environment=gen2",
        "--no-allow-unauthenticated",
        "--set-secrets=OPENROUTER_API_KEY=" + openrouter_secret + ",JWT_SECRET=" + jwt_secret,
        "--set-env-vars=ENVIRONMENT=staging,CORS_ORIGINS=https://halluciguard-ai.vercel.app",
        "--labels=scope=private-staging,review=55ef357,cleanup-within=24h",
    ]
    return {
        "execute": False,
        "prerequisites": [
            "Explicitly review and approve a bounded trial-credit scope; no hard monetary cap is enforced here",
            "Confirm candidate service does not already exist; preserve any existing service with this name",
            "Confirm image build passed and both secret versions and least-privilege runtime account already exist",
            "Confirm account has Artifact Registry read and access to only the required secret versions",
            "Confirm public allUsers and allAuthenticatedUsers invoker bindings are absent after deployment",
            "Keep production frontend/backend unchanged until actual candidate model and graph checks pass",
            "Review and remove only newly created staging resources within 24 hours",
        ],
        "command": command,
        "memory_durable": False,
        "production_traffic_change": False,
        "quality_certified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--runtime-service-account", required=True)
    parser.add_argument("--openrouter-secret-version", required=True)
    parser.add_argument("--jwt-secret-version", required=True)
    args = parser.parse_args()
    print(json.dumps(plan(args.image_digest, args.runtime_service_account,
                          args.openrouter_secret_version, args.jwt_secret_version), indent=2))
