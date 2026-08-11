#!/usr/bin/env python3
import sys, json, urllib.request, time, math

PROMPT = sys.argv[1] if len(sys.argv) > 1 else "What is 15 factorial?"
MAX_TOK = int(sys.argv[2]) if len(sys.argv) > 2 else 768
URL = "http://127.0.0.1:8000/v1/chat/completions"

def tool_math_factorial(args):
    n = int(args.get("number", args.get("n", 0)))
    return "The factorial of " + str(n) + " is " + str(math.factorial(n)) + "."

def tool_calculator(args):
    expr = args.get("expression", "")
    if expr:
        safe = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        try:
            return expr + " = " + str(eval(expr, {"__builtins__": {}}, safe))
        except:
            return "Could not evaluate: " + expr
    a = args.get("a", 0)
    b = args.get("b", 0)
    return str(a) + " x " + str(b) + " = " + str(a * b)

def tool_get_current_weather(args):
    loc = args.get("location", "unknown")
    return "Weather in " + loc + ": 22C, partly cloudy, humidity 65%."

def tool_search_web(args):
    q = args.get("query", "")
    return "Results for '" + q + "': [1] AI breakthrough 2026 [2] New model released [3] Industry report"

def tool_post_message(args):
    return "Posted to #" + args.get("channel", "general") + ": '" + args.get("text", "") + "'"

def tool_get_user_info(args):
    return "User " + str(args.get("user_id", "unknown")) + ": Name=Tejas, Role=Dev, Status=Active"

def tool_get_user_details(args):
    return tool_get_user_info(args)

def tool_get_doc(args):
    t = args.get("title", "").lower()
    if "open knowledge" in t or "okf" in t:
        return "Open Knowledge Format (OKF) is a specification by the Open Knowledge Foundation for packaging open data. It uses JSON-LD metadata, supports CSV/JSON/RDF, and includes provenance tracking and licensing."
    return "Document '" + t + "': comprehensive guide with definitions and best practices."

TOOL_REGISTRY = {
    "math.factorial": tool_math_factorial, "factorial": tool_math_factorial,
    "calculator": tool_calculator, "calculate": tool_calculator,
    "get_current_weather": tool_get_current_weather,
    "search_web": tool_search_web, "search": tool_search_web,
    "post_message": tool_post_message,
    "get_user_info": tool_get_user_info, "get_user_details": tool_get_user_details,
    "get_doc": tool_get_doc,
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

def safe_msg(resp):
    if not isinstance(resp, dict):
        return {}
    ch = resp.get("choices", [])
    if not ch or not isinstance(ch, list):
        return {}
    m = ch[0].get("message") if isinstance(ch[0], dict) else None
    return m if isinstance(m, dict) else {}

def extract_tools(msg):
    calls = []
    tcs = msg.get("tool_calls") if isinstance(msg, dict) else None
    if isinstance(tcs, list):
        for tc in tcs:
            if isinstance(tc, dict) and isinstance(tc.get("function"), dict):
                calls.append({"name": tc["function"].get("name"), "arguments": tc["function"].get("arguments", {})})
    content = msg.get("content", "") if isinstance(msg, dict) else ""
    if isinstance(content, str) and content.strip().startswith("["):
        try:
            for item in json.loads(content):
                if isinstance(item, dict) and "name" in item:
                    calls.append({"name": item["name"], "arguments": item.get("arguments", {})})
        except:
            pass
    return calls

print("=" * 70)
print("  OPTIMUZ FINAL VALIDATOR")
print("=" * 70)
print("Prompt: " + PROMPT)
print()

# Step 1
t0 = time.time()
resp1 = post(URL, {"messages": [{"role": "user", "content": PROMPT}], "max_tokens": MAX_TOK, "tool_choice": "auto", "temperature": 0.7})
lat1 = (time.time() - t0) * 1000
u1 = resp1.get("usage", {})
print("Step 1 | Tier: " + str(resp1.get("tier_used", "?")) + " | Latency: " + str(round(lat1, 1)) + "ms | Tokens: " + str(u1.get("total_tokens", "?")))
print()

msg1 = safe_msg(resp1)
tools = extract_tools(msg1)
direct = msg1.get("content", "") if isinstance(msg1.get("content"), str) else ""

if not tools:
    print("COMPLETE RESPONSE:")
    print(direct if direct else "(empty)")
else:
    print("Tools detected: " + str(len(tools)))
    results = []
    for tc in tools:
        name = tc.get("name", "?")
        args = tc.get("arguments", {})
        print("  -> " + name + "(" + json.dumps(args) + ")")
        fn = TOOL_REGISTRY.get(name)
        result = fn(args) if fn else "[Simulated result for " + name + "]"
        print("     Result: " + (result[:120] + "..." if len(result) > 120 else result))
        results.append(result)

    # Step 3: send as normal user message (avoids 422)
    followup = "Question: " + PROMPT + "\n\nTool results:\n"
    for r in results:
        followup += "- " + r + "\n"
    followup += "\nAnswer the question fully using the results above. Be complete."

    t0 = time.time()
    resp2 = post(URL, {"messages": [{"role": "user", "content": followup}], "max_tokens": MAX_TOK, "temperature": 0.7})
    lat2 = (time.time() - t0) * 1000
    u2 = resp2.get("usage", {})
    print()
    print("Step 2 | Tier: " + str(resp2.get("tier_used", "?")) + " | Latency: " + str(round(lat2, 1)) + "ms | Tokens: " + str(u2.get("total_tokens", "?")))
    print()
    final = safe_msg(resp2).get("content", "")
    print("COMPLETE FINAL RESPONSE:")
    print(final if final else "(empty)")

# Metrics
print()
print("=" * 70)
print("METRICS")
print("=" * 70)
cache = get("http://127.0.0.1:8000/v1/tools/cache")
print("Tool Cache: hits=" + str(cache.get("hits", "?")) + " misses=" + str(cache.get("misses", "?")) + " size=" + str(cache.get("size", "?")) + " rate=" + str(cache.get("hit_rate", "?")))
for name, port in [("tier1", 8081), ("tier2", 8082), ("tier3", 8083), ("tier-spec", 8084)]:
    s = get("http://127.0.0.1:" + str(port) + "/slots")
    if isinstance(s, list):
        c = sum(x.get("n_prompt_tokens_processed", 0) - x.get("n_prompt_tokens", 0) for x in s)
        print(name + ": " + str(len(s)) + " slots | " + str(c) + " cached tokens")
    else:
        print(name + ": " + str(s.get("error", "ok")))
schemas = resp1.get("tool_schemas_used", []) if isinstance(resp1, dict) else []
if schemas:
    print("Schemas: " + str(schemas))
print("=" * 70)
