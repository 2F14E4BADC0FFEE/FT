import os
from huggingface_hub import HfApi


def _get_hf_token():
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok and tok.strip():
            return tok.strip()
    except Exception:
        pass
    # Fallback to env (but we don't set dummy values beforehand)
    return os.environ.get("HF_TOKEN", "").strip()

def authenticate():
    """Approved thin wrapper: notebook cell 6 top-level auth sequence (validate + export).
    Sets HF_TOKEN and HUGGING_FACE_HUB_TOKEN in the environment (B8)."""
    HF_TOKEN = _get_hf_token()
    if not HF_TOKEN:
        raise RuntimeError(
            "❌ No token found.\n"
            "   - Accept the Gemma license: https://huggingface.co/google/gemma-4-E4B-it\n"
            "   - Add your HF_TOKEN to Kaggle Secrets and attach it."
        )
    # Quick sanity check
    if not HF_TOKEN.startswith("hf_"):
        raise RuntimeError("Token does not look like a user token (must start with 'hf_').")
    api = HfApi()
    try:
        user = api.whoami(token=HF_TOKEN)
        print(f"✅ Token valid – logged in as: {user['name']}")
    except Exception as e:
        raise RuntimeError(
            f"❌ Token validation still fails: {e}\n"
            "Double-check that you copied the token correctly and restarted the kernel."
        )
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    return HF_TOKEN
