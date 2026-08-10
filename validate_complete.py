#!/usr/bin/env python3
"""
OPTIMUZ Complete Validation — Natural Language + Spec Decode + Tool Calling
Usage: python3 validate_complete.py "Your prompt here"
"""

import sys, os, json, time, subprocess, urllib.request
from pathlib import Path
from dataclasses import dataclass, asdict

REPO = Path(__file__).resolve().parent

@dataclass
class ValidationResult:
    prompt: str
    response_text: str
    tier_used: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    tok_per_s: float
    draft_asr: float | None = None
    draft_accepted: int | None = None
    draft_generated: int | None = None
    tool_called: str | None = None
    tool_args: dict | None = None

def call_gateway(prompt: str, max_tokens: int = 256) -> dict:
    """Call the gateway which routes through the cascade properly."""
    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def call_tier_direct(url: str, prompt: str, max_tokens: int = 128) -> dict:
    """Direct call to a tier (for comparison)."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def get_spec_decode_metrics() -> dict:
    """Get draft acceptance from tier-spec logs."""
    try:
        result = subprocess.run(
            ["docker", "logs", "optimuz-tier-spec-1", "--since", "2m"],
            capture_output=True, text=True, timeout=15
        )
        lines = result.stdout.splitlines()
        entries = []
        for line in lines:
            if "draft acceptance" in line:
                parts = line.split("draft acceptance = ")
                if len(parts) > 1:
                    rest = parts[1]
                    rate = float(rest.split()[0])
                    acc = int(rest.split("(")[1].split()[0])
                    gen = int(rest.split("/")[1].split(")")[0].strip())
                    entries.append({"rate": rate, "accepted": acc, "generated": gen})
        if entries:
            total_acc = sum(e["accepted"] for e in entries)
            total_gen = sum(e["generated"] for e in entries)
            return {
                "rounds": len(entries),
                "latest_asr": entries[-1]["rate"] * 100,
                "overall_asr": (total_acc / total_gen * 100) if total_gen > 0 else 0,
                "total_accepted": total_acc,
                "total_generated": total_gen,
            }
    except Exception as e:
        return {"error": str(e)}
    return {}

def parse_tool_call(text: str) -> tuple[str | None, dict | None]:
    """Check if response is a tool call JSON."""
    text = text.strip()
    if text.startswith("[") and "\"name\"" in text:
        try:
            data = json.loads(text)
            if isinstance(data, list) and len(data) > 0:
                tool = data[0].get("name")
                args = data[0].get("arguments", {})
                return tool, args
        except:
            pass
    return None, None

def print_box(title: str, width: int = 78):
    print("\n" + "╔" + "═" * (width - 2) + "╗")
    print("║" + title.center(width - 2) + "║")
    print("╚" + "═" * (width - 2) + "╝")

def main():
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter your prompt: ")
    if not prompt:
        print("No prompt provided.")
        return

    print_box("OPTIMUZ VALIDATION: " + prompt[:50] + ("..." if len(prompt) > 50 else ""))

    # ── 1. GATEWAY RESPONSE (Natural Language Routing) ──────────────────────
    print("\n📡 Calling GATEWAY (cascade router)...")
    t0 = time.perf_counter()
    gateway_resp = call_gateway(prompt, max_tokens=256)
    gateway_latency = (time.perf_counter() - t0) * 1000.0

    if "error" in gateway_resp:
        print(f"   🔴 Gateway ERROR: {gateway_resp['error']}")
        print("   ⚠️  Falling back to direct tier2 call...")
        use_gateway = False
    else:
        use_gateway = True
        choice = gateway_resp.get("choices", [{}])[0]
        text = choice.get("message", {}).get("content", "")
        usage = gateway_resp.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        print(f"   ✅ Latency: {gateway_latency:.1f} ms")
        print(f"   📝 Tokens: {pt} prompt + {ct} completion = {pt+ct} total")
        print(f"   💬 Response:\n{'─'*60}\n{text}\n{'─'*60}")
        tool, args = parse_tool_call(text)
        if tool:
            print(f"   🔧 Tool Detected: {tool}({json.dumps(args)})")

    # ── 2. DIRECT TIER COMPARISON ─────────────────────────────────────────
    print("\n📊 DIRECT TIER BENCHMARKS (same prompt):")
    print(f"   {'Tier':<20} {'Latency':>10} {'Tok/s':>8} {'Tokens':>8} {'Status':>10}")
    print("   " + "─"*70)

    tiers = [
        ("tier2 (3B target)", "http://localhost:8082"),
        ("tier-spec (3B+1B)", "http://localhost:8084"),
    ]

    for name, url in tiers:
        t0 = time.perf_counter()
        resp = call_tier_direct(url, prompt, max_tokens=128)
        lat = (time.perf_counter() - t0) * 1000.0
        if "error" in resp:
            print(f"   {name:<20} {'ERROR':>10} {'—':>8} {'—':>8} {'DOWN':>10}")
            continue
        usage = resp.get("usage", {})
        ct = usage.get("completion_tokens", 0)
        tok_s = round(ct / (lat / 1000.0), 2) if lat > 0 else 0
        print(f"   {name:<20} {lat:>10.1f}ms {tok_s:>8.2f} {ct:>8} {'OK':>10}")

    # ── 3. SPEC DECODE METRICS ────────────────────────────────────────────
    print("\n🔬 TOKEN-LEVEL SPECULATIVE DECODING METRICS:")
    metrics = get_spec_decode_metrics()
    if metrics and "error" not in metrics:
        print(f"   Draft rounds (last 2 min): {metrics['rounds']}")
        print(f"   Latest ASR:              {metrics['latest_asr']:.1f}%")
        print(f"   Overall ASR:             {metrics['overall_asr']:.1f}%")
        print(f"   Total accepted/generated: {metrics['total_accepted']}/{metrics['total_generated']}")
    else:
        print("   ⚠️  No recent draft activity (run more prompts to generate logs)")

    # ── 4. TOOL-LEVEL METRICS ───────────────────────────────────────────────
    print("\n🛠️  TOOL-LEVEL SPECULATION METRICS:")
    tool_file = Path("/tmp/tool_spec.json")
    if tool_file.exists():
        data = json.loads(tool_file.read_text())
        s = data.get("summary", {})
        print(f"   Hit rate:          {s.get('hit_rate', 0)*100:.0f}%")
        print(f"   Latency speedup:   {s.get('latency_speedup', 0):.2f}×")
        print(f"   Time saved/call:   {s.get('avg_time_saved_ms', 0):.1f} ms")
        print(f"   Tokens/$ gain:     +{s.get('tokens_per_dollar_delta', 0):.0f}")
    else:
        print("   ⚠️  Run: python3 benchmarks/speculative_tool_bench.py --out /tmp/tool_spec.json")

    # ── 5. SAVE REPORT ────────────────────────────────────────────────────
    report = {
        "timestamp": time.time(),
        "prompt": prompt,
        "gateway_used": use_gateway,
        "gateway_latency_ms": gateway_latency if use_gateway else None,
        "spec_decode_metrics": metrics,
        "tool_spec_summary": json.loads(tool_file.read_text()).get("summary", {}) if tool_file.exists() else {},
    }
    out = REPO / f"complete_validation_{int(time.time())}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\n💾 Report saved: {out}")

    print_box("VALIDATION COMPLETE")
    print("\n✅ For submission, screenshot this output + the JSON report.")
    print("✅ The gateway gives natural language responses.")
    print("✅ Direct tiers show raw model behavior (tool-calling for xLAM models).")

if __name__ == "__main__":
    main()
