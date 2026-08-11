#!/usr/bin/env python3
"""
OPTIMUZ MCP Tool Validator
Usage: python3 validate_mcp.py "Your prompt here" [max_tokens]
"""
import sys, json, urllib.request, time

PROMPT  = sys.argv[1] if len(sys.argv) > 1 else "What is 15 factorial?"
MAX_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 512
URL     = "http://127.0.0.1:8000/v1/chat/completions"

def post(url, payload, timeout=60):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def get(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}

# ── 1. Send prompt with tool_choice=auto ──
payload = {
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": MAX_TOK,
    "tool_choice": "auto"
}

t0   = time.time()
resp = post(URL, payload)
lat  = (time.time() - t0) * 1000

msg      = resp.get("choices", [{}])[0].get("message", {})
content  = msg.get("content", "")
tools    = msg.get("tool_calls", [])
usage    = resp.get("usage", {})
metrics  = resp.get("metrics", {})
tier     = resp.get("tier_used", "unknown")
model    = resp.get("model", "unknown")
schemas  = resp.get("tool_schemas_used", [])

# ── 2. Real ToolOutputCache metrics ──
cache = get("http://127.0.0.1:8000/v1/tools/cache")

# ── 3. Print full report ──
print("=" * 70)
print("  OPTIMUZ MCP TOOL VALIDATION REPORT")
print("=" * 70)
print(f"Prompt          : {PROMPT}")
print(f"Max Tokens      : {MAX_TOK}")
print(f"Tier Used       : {tier}")
print(f"Model           : {model}")
print(f"Client Latency  : {lat:.2f} ms")
print(f"Gateway Latency : {metrics.get('latency_ms', 'N/A')} ms")
print()

print("--- FULL RESPONSE ---")
print(content if content else "(no text content)")
print()

print("--- TOOL CALLS EMITTED ---")
if tools:
    for i, t in enumerate(tools, 1):
        fn = t.get("function", {})
        print(f"  [{i}] Name: {fn.get('name')}")
        print(f"      Args: {json.dumps(fn.get('arguments', {}), indent=8)}")
else:
    print("  None")
print()

print("--- TOOL SCHEMAS CONSIDERED ---")
for s in schemas:
    print(f"  • {s}")
print()

print("--- TOKEN USAGE ---")
print(f"  Prompt tokens     : {usage.get('prompt_tokens', 'N/A')}")
print(f"  Completion tokens : {usage.get('completion_tokens', 'N/A')}")
print(f"  Total tokens      : {usage.get('total_tokens', 'N/A')}")
print()

print("--- TOOL OUTPUT CACHE (REAL) ---")
print(f"  Hits      : {cache.get('hits', 'N/A')}")
print(f"  Misses    : {cache.get('misses', 'N/A')}")
print(f"  Size      : {cache.get('size', 'N/A')}")
print(f"  Hit Rate  : {cache.get('hit_rate', 'N/A')}")
print(f"  Top Keys  : {cache.get('top_keys', [])}")
print()

print("--- GATEWAY METRICS ---")
print(f"  Cache hit flag              : {metrics.get('cache_hit', 'N/A')}")
print(f"  Speculative hit flag        : {resp.get('speculative_hit', 'N/A')}")
print(f"  Speculative latency saved   : {resp.get('speculative_latency_saved_ms', 'N/A')} ms")
print(f"  Degraded flag               : {metrics.get('degraded', 'N/A')}")
print(f"  Thinking token cap          : {resp.get('thinking_token_cap', 'N/A')}")
print("=" * 70)
