VERIFIER_VERSION = "2.0.0"
API_VERSION = "v2"
SCHEMA_VERSION = "2.0"
MODEL_VERSION = "1.0"

def get_version_info() -> dict:
    return {
        "verifier_version": VERIFIER_VERSION,
        "api_version": API_VERSION,
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
    }
