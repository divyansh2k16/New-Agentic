"""
Anthropic API Key Diagnostic Script

Usage:
    python scripts/test_api_key.py

Tests:
  1. Key is set and has the right format
  2. API is reachable (network / proxy check)
  3. Key is valid and not expired (lightweight models.list call)
  4. Account has usable credit (small claude-haiku call)
  5. The model names configured in settings are accessible
"""
import os
import sys
from pathlib import Path

# Allow running from anywhere inside the project
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Try loading .env ──────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; rely on shell env


def _ok(msg):   print(f"  ✓  {msg}")
def _fail(msg): print(f"  ✗  {msg}")
def _info(msg): print(f"     {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Key presence & format
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 1. API Key Check ─────────────────────────────────────────────────")
api_key = os.getenv("ANTHROPIC_API_KEY", "")

if not api_key:
    _fail("ANTHROPIC_API_KEY is not set in environment or .env file")
    sys.exit(1)

if not api_key.startswith("sk-ant-"):
    _fail(f"Key format looks wrong — expected 'sk-ant-...' but got '{api_key[:12]}...'")
    _info("Go to https://console.anthropic.com/settings/keys and copy the full key")
else:
    _ok(f"Key found and format looks correct: {api_key[:16]}...{api_key[-4:]}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SDK import
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 2. SDK Import ────────────────────────────────────────────────────")
try:
    import anthropic
    _ok(f"anthropic SDK imported (version {anthropic.__version__})")
except ImportError:
    _fail("anthropic package not installed — run: pip install anthropic")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Network reachability (HEAD request to API)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 3. Network / Proxy Check ─────────────────────────────────────────")
try:
    import httpx
    r = httpx.head("https://api.anthropic.com", timeout=8)
    _ok(f"api.anthropic.com is reachable (HTTP {r.status_code})")
except Exception as e:
    _fail(f"Cannot reach api.anthropic.com: {e}")
    _info("Check your internet connection or corporate proxy settings")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Key validity — list models (no credit needed, auth-only check)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 4. Key Validity (no credit needed) ───────────────────────────────")
client = anthropic.Anthropic(api_key=api_key)

try:
    models = client.models.list()
    model_ids = [m.id for m in models.data]
    _ok(f"Key is valid — {len(model_ids)} model(s) accessible")
    for mid in model_ids[:5]:
        _info(f"  · {mid}")
    if len(model_ids) > 5:
        _info(f"  ... and {len(model_ids) - 5} more")
except anthropic.AuthenticationError as e:
    _fail(f"Authentication failed — key is invalid or revoked: {e}")
    _info("Generate a new key at https://console.anthropic.com/settings/keys")
    sys.exit(1)
except Exception as e:
    _fail(f"Unexpected error checking key: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Credit check — tiny Haiku call (cheapest possible)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── 5. Credit / Billing Check (minimal Haiku call) ───────────────────")
try:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{"role": "user", "content": "Say OK"}],
    )
    reply = response.content[0].text.strip()
    usage = response.usage
    _ok(f"Call succeeded — reply: '{reply}'")
    _info(f"Tokens used: {usage.input_tokens} in / {usage.output_tokens} out")
    _info("Your key has usable credit.")
except anthropic.BadRequestError as e:
    if "credit balance is too low" in str(e).lower() or "402" in str(e):
        _fail("Credit balance is too low — API call rejected (HTTP 400/402)")
        _info("")
        _info("── Why this happens despite showing $5 in the console ──")
        _info("")
        _info("  (a) FREE TIER EXPIRY: Trial credits expire. Even if the dashboard")
        _info("      shows $5, expired credits can't be used. Add a payment method")
        _info("      and purchase prepaid credits to get spendable balance.")
        _info("")
        _info("  (b) WRONG WORKSPACE: Your key belongs to workspace A but the")
        _info("      $5 is in workspace B. Keys are workspace-scoped.")
        _info("      Check: console.anthropic.com → (top-left dropdown) → workspace")
        _info("")
        _info("  (c) PAYMENT METHOD REQUIRED: Some accounts need a verified card")
        _info("      on file before credits are usable, even prepaid ones.")
        _info("")
        _info("  (d) INVOICE OVERDUE: If you have an invoice-based plan with an")
        _info("      outstanding balance the API is blocked until it is paid.")
        _info("")
        _info("  Fix: https://console.anthropic.com/settings/billing")
        _info("       → 'Add payment method' or 'Buy credits'")
    else:
        _fail(f"Bad request error: {e}")
except anthropic.AuthenticationError as e:
    _fail(f"Auth error on credit check: {e}")
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
        "primary_llm_model": cfg.primary_llm_model,
        "fast_llm_model": cfg.fast_llm_model,
    }
    available_ids = set()
    try:
        available_ids = {m.id for m in client.models.list().data}
    except Exception:
        pass

    for setting, model_id in configured_models.items():
        if available_ids and model_id not in available_ids:
            _fail(f"{setting} = '{model_id}' — NOT found in your account's model list")
            _info("Check https://docs.anthropic.com/en/docs/about-claude/models")
        else:
            _ok(f"{setting} = '{model_id}'")
except Exception as e:
    _info(f"Could not check settings models: {e}")


print("\n─────────────────────────────────────────────────────────────────────")
print("Done. Fix any ✗ items above, then re-run to confirm.\n")
