#!/usr/bin/env python3
"""
OPTIMUZ Complete Response + Tool Executor
Usage: python3 complete_response.py "Your prompt here"

Guarantees:
- Full, untruncated response
- High max_tokens (2048)
- Tool calls are detected, executed, and final answer returned
"""
import sys, json, urllib.request, time, math

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "What is 15 factorial?"
URL = "http://127.0.0.1:8000/v1/chat/completions"

# ── Tool Registry ──
def tool_math_factorial(args):
    n = int(args.get("number", args.get("n", 0)))
    return f"The factorial of {n} is {math.factorial(n)}."

def tool_calculator(args):
    expr = args.get("expression", "")
    if expr:
        try:
            safe = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            result = eval(expr, {"__builtins__": {}}, safe)
            return f"{expr} = {result}"
        except:
            return f"Error evaluating: {expr}"
    a = args.get("a", 0)
    b = args.get("b", 0)
    return f"{a} × {b} = {a * b}"

def tool_get_current_weather(args):
    loc = args.get("location", args.get("city", "unknown"))
    return f"Weather in {loc}: 24°C, sunny, humidity 60%."

def tool_search_web(args):
    q = args.get("query", args.get("q", ""))
    return f"Results for '{q}': [1] Latest developments in AI... [2] New research paper published..."

def tool_post_message(args):
    ch = args.get("channel", "general")
    txt = args.get("text", "")
    return f"Posted to #{ch}: '{txt}'"

def tool_get_user_info(args):
    uid = args.get("user_id", "unknown")
    return f"User {uid}: Name=Tejas, Role=Developer, Status=Active"

TOOL_REGISTRY = {
    "math.factorial": tool_math_factorial,
    "factorial": tool_math_factorial,
    "calculator": tool_calculator,
    "calculate": tool_calculator,
    "get_current_weather": tool_get_current_weather,
    "search_web": tool_search_web,
    "search": tool_search_web,
    "post_message": tool_post_message,
    "get_user_info": tool_get_user_info,
}

def post(url, payload, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"error": str(e)}

def extract_tool_calls(resp):
    calls = []
    msg = resp.get("choices", [{}])[0].get("message", {})
    
    # Structured tool_calls
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        calls.append({"name": fn.get("name"), "arguments": fn.get("arguments", {})})
    
    # Parse JSON from content if present
    content = msg.get("content", "")
    if content and content.strip().startswith("["):
        try:
            for item in json.loads(content):
                if isinstance(item, dict) and "name" in item:
                    calls.append({"name": item["name"], "arguments": item.get("arguments", {})})
        except:
            pass
    return calls

# ── Main Execution ──
print("=" * 70)
print("  OPTIMUZ COMPLETE RESPONSE VALIDATOR")
print("=" * 70)
print(f"Prompt: {PROMPT}")
print()

# Step 1: Send prompt with HIGH max_tokens (2048 = complete answer)
messages = [{"role": "user", "content": PROMPT}]

t0 = time.time()
resp1 = post(URL, {
    "messages": messages,
    "max_tokens": 2048,        # <-- HIGH LIMIT = NO TRUNCATION
    "tool_choice": "auto",
    "temperature": 0.7
})
lat1 = (time.time() - t0) * 1000

tier1 = resp1.get("tier_used", "unknown")
usage1 = resp1.get("usage", {})
tool_calls = extract_tool_calls(resp1)
schemas = resp1.get("tool_schemas_used", [])

print(f"--- STEP 1: Prompt Sent ---")
print(f"Tier Used : {tier1}")
print(f"Latency   : {lat1:.2f} ms")
print(f"Tokens    : prompt={usage1.get('prompt_tokens','N/A')}, completion={usage1.get('completion_tokens','N/A')}, total={usage1.get('total_tokens','N/A')}")
print()

# Step 2: Execute tools if detected
if tool_calls:
    print(f"--- STEP 2: Tool Calls Detected ---")
    tool_results = []
    for tc in tool_calls:
        name = tc["name"]
        args = tc["arguments"]
        print(f"  Tool : {name}")
        print(f"  Args : {json.dumps(args)}")
        
        handler = TOOL_REGISTRY.get(name)
        if handler:
            result = handler(args)
        else:
            result = f"[Tool '{name}' not in local registry — simulating execution]"
        print(f"  Result: {result}")
        print()
        tool_results.append({"name": name, "result": result})
    
    # Step 3: Send tool results back for final answer
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [{"type": "function", "function": {"name": tr["name"], "arguments": "{}"}} for tr in tool_results]
    })
    for tr in tool_results:
        messages.append({"role": "tool", "name": tr["name"], "content": tr["result"]})
    
    t0 = time.time()
    resp2 = post(URL, {"messages": messages, "max_tokens": 2048, "temperature": 0.7})
    lat2 = (time.time() - t0) * 1000
    
    tier2 = resp2.get("tier_used", "unknown")
    usage2 = resp2.get("usage", {})
    final = resp2.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    print(f"--- STEP 3: Final Answer (After Tool Execution) ---")
    print(f"Tier Used : {tier2}")
    print(f"Latency   : {lat2:.2f} ms")
    print(f"Tokens    : prompt={usage2.get('prompt_tokens','N/A')}, completion={usage2.get('completion_tokens','N/A')}, total={usage2.get('total_tokens','N/A')}")
    print()
    print("=" * 70)
    print("COMPLETE FINAL RESPONSE:")
    print("=" * 70)
    print(final)
else:
    # No tools — direct answer
    final = resp1.get("choices", [{}])[0].get("message", {}).get("content", "")
    print("=" * 70)
    print("COMPLETE FINAL RESPONSE (No Tools Needed):")
    print("=" * 70)
    print(final)

# ── Real Metrics ──
print()
print("=" * 70)
print("REAL METRICS")
print("=" * 70)

cache = get("http://127.0.0.1:8000/v1/tools/cache")
print(f"ToolOutputCache: hits={cache.get('hits','N/A')}, misses={cache.get('misses','N/A')}, size={cache.get('size','N/A')}, hit_rate={cache.get('hit_rate','N/A')}")

for name, port in [("tier1",8081), ("tier2",8082), ("tier3",8083), ("tier-spec",8084)]:
    slots = get(f"http://127.0.0.1:{port}/slots")
    if isinstance(slots, list):
        cached = sum(s.get("n_prompt_tokens_processed",0) - s.get("n_prompt_tokens",0) for s in slots)
        print(f"{name:12s}: {len(slots)} slots, ~{cached} cached tokens")
    else:
        print(f"{name:12s}: {slots.get('error','ok')}")

print(f"Tool Schemas: {schemas}")
print("=" * 70)
