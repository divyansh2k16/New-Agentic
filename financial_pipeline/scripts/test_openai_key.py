"""
OpenAI API Key Diagnostic Script

Usage:
    python scripts/test_openai_key.py

Tests:
  1. Key is set and has the right format  (sk-proj-... or sk-...)
  2. API is reachable (network / proxy check)
  3. Key is valid (list models — no credit needed)
  4. Account has usable quota (tiny gpt-4o-mini call)
  5. The model names configured in settings are accessible
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # rely on shell env


def _ok(msg):   print(f"  ✓  {msg}")
def _fail(msg): print(f"  ✗  {msg}")
def _info(msg): print(f"     {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Key presence & format
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 1. API Key Check ─────────────────────────────────────────────────")
api_key = os.getenv("OPENAI_API_KEY", "")

if not api_key:
    _fail("OPENAI_API_KEY is not set in environment or .env file")
    sys.exit(1)

if not (api_key.startswith("sk-proj-") or api_key.startswith("sk-")):
    _fail(f"Key format looks wrong — expected 'sk-...' but got '{api_key[:12]}...'")
    _info("Go to https://platform.openai.com/api-keys and copy the full key")
elif len(api_key) < 40:
    _fail(f"Key looks too short ({len(api_key)} chars) — may be truncated or a placeholder")
    _info("Make sure you pasted the complete key from https://platform.openai.com/api-keys")
else:
    _ok(f"Key found and format looks correct: {api_key[:16]}...{api_key[-4:]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SDK import
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2. SDK Import ────────────────────────────────────────────────────")
try:
    import openai
    _ok(f"openai SDK imported (version {openai.__version__})")
except ImportError:
    _fail("openai package not installed — run: pip install openai")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Network reachability
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 3. Network / Proxy Check ─────────────────────────────────────────")
try:
    import httpx
    r = httpx.head("https://api.openai.com", timeout=8)
    _ok(f"api.openai.com is reachable (HTTP {r.status_code})")
except Exception as e:
    _fail(f"Cannot reach api.openai.com: {e}")
    _info("Check your internet connection or corporate proxy settings")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Key validity — list models (no credit needed, auth-only check)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4. Key Validity (no quota needed) ────────────────────────────────")
client = openai.OpenAI(api_key=api_key)

try:
    models = client.models.list()
    model_ids = sorted([m.id for m in models.data])
    _ok(f"Key is valid — {len(model_ids)} model(s) accessible")
    gpt_models = [m for m in model_ids if m.startswith("gpt-")]
    for mid in gpt_models[:5]:
        _info(f"  · {mid}")
    if len(gpt_models) > 5:
        _info(f"  ... and {len(gpt_models) - 5} more GPT models")
except openai.AuthenticationError as e:
    _fail(f"Authentication failed — key is invalid or revoked")
    _info(f"Error: {e}")
    _info("")
    _info("── Common causes ─────────────────────────────────────────────────")
    _info("  (a) WRONG KEY: The key in .env is a placeholder/example, not real.")
    _info("      Get a real key at https://platform.openai.com/api-keys")
    _info("")
    _info("  (b) KEY REVOKED: The key was deleted or rotated.")
    _info("      Create a new one at https://platform.openai.com/api-keys")
    _info("")
    _info("  (c) ORG MISMATCH: The key belongs to a different organisation.")
    _info("      Check platform.openai.com → Settings → Organisation")
    sys.exit(1)
except openai.PermissionDeniedError as e:
    _fail(f"Permission denied: {e}")
    _info("Your key may lack the required scopes — check key permissions at platform.openai.com")
    sys.exit(1)
except Exception as e:
    _fail(f"Unexpected error checking key: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Quota / billing check — tiny gpt-4o-mini call (cheapest possible)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5. Quota / Billing Check (minimal gpt-4o-mini call) ──────────────")
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=10,
        messages=[{"role": "user", "content": "Say OK"}],
    )
    reply = response.choices[0].message.content.strip()
    usage = response.usage
    _ok(f"Call succeeded — reply: '{reply}'")
    _info(f"Tokens used: {usage.prompt_tokens} in / {usage.completion_tokens} out")
    _info("Your key has usable quota.")
except openai.RateLimitError as e:
    _fail("Rate limit or quota exceeded")
    _info(f"Error: {e}")
    _info("")
    _info("── Why this happens ──────────────────────────────────────────────")
    _info("  (a) FREE TIER EXHAUSTED: $5 trial credit is used up.")
    _info("      Add a payment method at https://platform.openai.com/settings/billing")
    _info("")
    _info("  (b) RATE LIMIT: Too many requests per minute for your tier.")
    _info("      Wait a minute and retry, or upgrade your usage tier.")
    _info("")
    _info("  (c) HARD LIMIT REACHED: You set a monthly spend cap that was hit.")
    _info("      Check https://platform.openai.com/settings/billing/limits")
except openai.AuthenticationError as e:
    _fail(f"Auth error on quota check: {e}")
except Exception as e:
    _fail(f"Unexpected error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Verify the model names used in settings
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 6. Settings Model Check ──────────────────────────────────────────")
try:
    from config.settings import get_settings
    cfg = get_settings()
    configured_models = {
        "openai_primary_model": cfg.openai_primary_model,
        "openai_fast_model":    cfg.openai_fast_model,
    }
    available_ids = set()
    try:
        available_ids = {m.id for m in client.models.list().data}
    except Exception:
        pass

    for setting, model_id in configured_models.items():
        if available_ids and model_id not in available_ids:
            _fail(f"{setting} = '{model_id}' — NOT in your account's model list")
            _info("Check https://platform.openai.com/docs/models")
        else:
            _ok(f"{setting} = '{model_id}'")
except Exception as e:
    _info(f"Could not check settings models: {e}")


print("\n─────────────────────────────────────────────────────────────────────")
print("Done. Fix any ✗ items above, then re-run to confirm.\n")
