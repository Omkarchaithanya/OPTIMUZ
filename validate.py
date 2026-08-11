#!/usr/bin/env python3
import sys, json, urllib.request, time

PROMPT    = sys.argv[1] if len(sys.argv) > 1 else "Hello"
MAX_TOK   = int(sys.argv[2]) if len(sys.argv) > 2 else 512
GATEWAY   = "http://127.0.0.1:8000/v1/chat/completions"

def fetch_json(url, timeout=5, method="GET", data=None):
    try:
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type":"application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}

def fetch_text(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()
    except Exception as e:
        return f"Error: {e}"

# ── 1. GATEWAY INFERENCE ──
payload = json.dumps({"messages":[{"role":"user","content":PROMPT}],"max_tokens":MAX_TOK}).encode()
t0 = time.time()
resp = fetch_json(GATEWAY, timeout=60, method="POST", data=payload)
client_ms = (time.time() - t0) * 1000

tier        = resp.get("tier_used", "unknown")
model       = resp.get("model", "unknown")
content     = resp.get("choices",[{}])[0].get("message",{}).get("content","")
usage       = resp.get("usage",{})
metrics     = resp.get("metrics",{})
tool_calls  = resp.get("choices",[{}])[0].get("message",{}).get("tool_calls")
schemas     = resp.get("tool_schemas_used",[])
spec_hit    = resp.get("speculative_hit", False)
spec_saved  = resp.get("speculative_latency_saved_ms", 0)

# ── 2. TOOL OUTPUT CACHE (real) ──
tool_cache = fetch_json("http://127.0.0.1:8000/v1/tools/cache")

# ── 3. KV CACHE / SLOTS per tier (real) ──
tiers = {"tier1":8081, "tier2":8082, "tier3":8083, "tier-spec":8084}
slots = {name: fetch_json(f"http://127.0.0.1:{port}/slots") for name,port in tiers.items()}

# ── 4. PROMETHEUS METRICS (real) ──
prom = fetch_text("http://127.0.0.1:8000/metrics")
prom_lines = [l for l in prom.split("\n") if any(k in l for k in ["nsa_kv","nsa_tool","nsa_spec","nsa_tier","nsa_latency"])][:25]

# ── 5. SPECULATIVE DECODE direct via tier-spec ──
spec_payload = json.dumps({"messages":[{"role":"user","content":PROMPT}],"max_tokens":MAX_TOK}).encode()
t_spec0 = time.time()
spec_resp = fetch_json("http://127.0.0.1:8084/v1/chat/completions", timeout=60, method="POST", data=spec_payload)
spec_ms = (time.time() - t_spec0) * 1000

# ── PRINT REPORT ──
print("="*70)
print("  OPTIMUZ FULL VALIDATION REPORT")
print("="*70)
print(f"Prompt        : {PROMPT}")
print(f"Client Latency: {client_ms:.2f} ms")
print()

print("--- INFERENCE RESULT ---")
print(f"Tier Used     : {tier}")
print(f"Model         : {model}")
print(f"Response      :\n{content}")
print()

print("--- TOKEN METRICS ---")
print(json.dumps(usage, indent=2))
print()

print("--- GATEWAY METRICS ---")
print(json.dumps({
    "gateway_internal_latency_ms": metrics.get("latency_ms"),
    "cache_hit_flag": metrics.get("cache_hit"),
    "speculative_hit_flag": spec_hit,
    "speculative_latency_saved_ms": spec_saved,
    "degraded_flag": metrics.get("degraded"),
    "thinking_token_cap": resp.get("thinking_token_cap")
}, indent=2))
print()

print("--- TOOL CALLING ---")
print(f"Tool Schemas Considered : {schemas}")
print(f"Tool Calls Emitted      : {json.dumps(tool_calls, indent=2) if tool_calls else 'None'}")
print()

print("--- TOOL OUTPUT CACHE (REAL) ---")
print(json.dumps(tool_cache, indent=2))
print()

print("--- KV CACHE / SLOTS PER TIER (REAL) ---")
for name, data in slots.items():
    if "error" in data:
        print(f"  {name:12s}: {data['error']}")
        continue
    print(f"  {name}:")
    if isinstance(data, list) and data:
        for s in data:
            cached = s.get("n_prompt_tokens_processed",0) - s.get("n_prompt_tokens",0)
            print(f"    Slot {s.get('id')} | state={s.get('state','?')} | prompt={s.get('n_prompt_tokens',0)} | processed={s.get('n_prompt_tokens_processed',0)} | cached={cached} | decode={s.get('n_decode',0)}")
    else:
        print(f"    (no active slots or empty response: {json.dumps(data)[:120]})")
print()

print("--- SPECULATIVE DECODE (tier-spec direct) ---")
print(f"  Latency     : {spec_ms:.2f} ms")
print(f"  Tokens used : {json.dumps(spec_resp.get('usage',{}), indent=2)}")
print(f"  Content preview (first 200 chars):")
print(f"  {spec_resp.get('choices',[{}])[0].get('message',{}).get('content','')[:200]}")
print()

print("--- PROMETHEUS / GATEWAY METRICS ---")
if prom_lines:
    for line in prom_lines:
        print(f"  {line}")
else:
    print("  (No nsa_* metrics on gateway /metrics — check Prometheus container)")
print("="*70)
