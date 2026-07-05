#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Call SCNet OpenAI-compatible Chat API to create structured LLM prior JSONL.

Official API discovered from SCNet console docs:
  base_url: https://api.scnet.cn/api/llm/v1
  endpoint: /chat/completions
  auth: Authorization: Bearer <API Key>

This script never stores the API key. Set it in the environment:
  PowerShell:
    $env:SCNET_API_KEY="..."
  Bash:
    export SCNET_API_KEY="..."

Recommended first pass for this project:
  python 21_call_scnet_deepseek_priors.py \
    --input-jsonl outputs_addh_llm_element_priors/llm_prior_prompts.jsonl \
    --output-jsonl outputs_addh_llm_element_priors/llm_prior_scnet_deepseek_v4_pro_addhout.jsonl \
    --id-regex "^(CeO2|ZnO)-" \
    --model DeepSeek-V4-Pro
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional


NUMERIC_FIELDS = [
    "llm_prior_h_ads_eV_guess",
    "llm_expected_rank_score",
    "llm_h_binding_strength_score",
    "llm_oxygen_affinity_score",
    "llm_oxide_reducibility_score",
    "llm_charge_compensation_complexity",
    "llm_host_similarity_to_training",
    "llm_confidence",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Call SCNet DeepSeek model for structured AddH-out LLM priors.")
    ap.add_argument("--input-jsonl", default="outputs_addh_llm_element_priors/llm_prior_prompts.jsonl")
    ap.add_argument("--output-jsonl", default="outputs_addh_llm_element_priors/llm_prior_scnet_deepseek_v4_pro_addhout.jsonl")
    ap.add_argument("--failed-jsonl", default=None)
    ap.add_argument("--base-url", default="https://api.scnet.cn/api/llm/v1")
    ap.add_argument("--model", default="DeepSeek-V4-Pro")
    ap.add_argument("--id-regex", default="^(CeO2|ZnO)-", help="Only call records whose custom_id matches this regex.")
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=900)
    ap.add_argument("--reasoning-effort", default="high", choices=["high", "max"])
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--no-response-format", action="store_true", help="Disable OpenAI response_format json_object.")
    ap.add_argument(
        "--prompt-style",
        default="original",
        choices=["original", "compact"],
        help="Use original generated prompt or a shorter single-user JSON prompt for unstable API responses.",
    )
    ap.add_argument("--no-system-message", action="store_true", help="Drop system messages from original prompts.")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--sleep", type=float, default=0.8)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                rows.append(json.loads(raw))
    return rows


def append_jsonl(path: Path, rec: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_done_ids(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            key = str(rec.get("id") or rec.get("custom_id") or "").strip()
            if key:
                done.add(key)
    return done


def extract_line_value(text: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}:\s*(.+?)\s*$", text or "", flags=re.M)
    return m.group(1).strip() if m else ""


def prompt_user_text(prompt_rec: Dict) -> str:
    body = prompt_rec.get("body") or {}
    for m in body.get("messages") or []:
        if str(m.get("role", "")).lower() == "user":
            return str(m.get("content", ""))
    return ""


def compact_messages(prompt_rec: Dict) -> List[Dict[str, str]]:
    cid = str(prompt_rec.get("custom_id") or prompt_rec.get("id") or "").strip()
    text = prompt_user_text(prompt_rec)
    material = extract_line_value(text, "material/host")
    dopant = extract_line_value(text, "dopant")
    miller = extract_line_value(text, "miller")
    non_h_elements = extract_line_value(text, "non_H_elements_in_CONTCAR")
    dopant_props = extract_line_value(text, "dopant_properties")

    schema = {
        "id": cid,
        "material": material,
        "dopant": dopant,
        "llm_prior_h_ads_eV_guess": "number or null",
        "llm_expected_rank_score": "number -3 strong adsorption to +3 weak adsorption",
        "llm_h_binding_strength_score": "number -3 weak to +3 strong",
        "llm_oxygen_affinity_score": "number 0 to 5",
        "llm_oxide_reducibility_score": "number 0 to 5",
        "llm_charge_compensation_complexity": "number 0 to 5",
        "llm_host_similarity_to_training": "number 0 to 1",
        "llm_confidence": "number 0 to 1",
        "rationale_short": "one short sentence",
        "sources": ["short source names, DOI, or empty list"],
    }
    content = (
        "Return exactly one valid JSON object and no markdown.\n"
        "Estimate qualitative priors for H adsorption on this doped oxide surface using periodic trends, "
        "oxide chemistry, reducibility, charge compensation, and dopant/host similarity only. "
        "Do not use hidden labels or dataset values. Use null only for the adsorption-energy guess if you are unsure.\n\n"
        f"sample_id: {cid}\n"
        f"material/host: {material}\n"
        f"dopant: {dopant}\n"
        f"miller: {miller}\n"
        f"non_H_elements_in_CONTCAR: {non_h_elements}\n"
        f"dopant_properties: {dopant_props}\n\n"
        "Required JSON keys and value ranges:\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )
    return [{"role": "user", "content": content}]


def extract_messages(prompt_rec: Dict, drop_system: bool = False) -> List[Dict[str, str]]:
    body = prompt_rec.get("body") or {}
    messages = body.get("messages") or []
    out = []
    for m in messages:
        role = str(m.get("role", "user"))
        if drop_system and role.lower() == "system":
            continue
        content = str(m.get("content", ""))
        if content:
            out.append({"role": role, "content": content})
    if not out:
        raise ValueError(f"prompt record has no messages: {prompt_rec.get('custom_id')}")
    return out


def make_payload(prompt_rec: Dict, args: argparse.Namespace) -> Dict:
    if args.prompt_style == "compact":
        messages = compact_messages(prompt_rec)
    else:
        messages = extract_messages(prompt_rec, drop_system=args.no_system_message)
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.disable_thinking:
        payload["enable_thinking"] = False
    else:
        payload["reasoning_effort"] = args.reasoning_effort
        payload["enable_thinking"] = True
    if not args.no_response_format:
        payload["response_format"] = {"type": "json_object"}
    return payload


def strip_json_fence(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s, flags=re.I).strip()
        s = re.sub(r"```$", "", s).strip()
    return s


def parse_model_json(text: str) -> Dict:
    clean = strip_json_fence(text)
    try:
        return json.loads(clean)
    except Exception:
        # Fallback: recover first JSON object in a verbose answer.
        m = re.search(r"\{.*\}", clean, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


def response_message_text(choice: Dict) -> str:
    msg = choice.get("message") or {}
    for key in ("content", "reasoning_content", "reasoning", "text"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val
    val = choice.get("text")
    if isinstance(val, str) and val.strip():
        return val
    return ""


def coerce_prior(custom_id: str, parsed: Dict, raw_text: str, response: Dict) -> Dict:
    rec: Dict = {
        "id": str(parsed.get("id") or custom_id),
        "custom_id": custom_id,
        "material": parsed.get("material"),
        "dopant": parsed.get("dopant"),
        "rationale_short": parsed.get("rationale_short", ""),
        "sources": parsed.get("sources", []),
        "scnet_model": response.get("model"),
        "scnet_response_id": response.get("id"),
        "scnet_usage": response.get("usage", {}),
    }
    for c in NUMERIC_FIELDS:
        val = parsed.get(c, None)
        try:
            rec[c] = None if val is None else float(val)
        except Exception:
            rec[c] = None
    rec["raw_content"] = raw_text
    return rec


def call_scnet(payload: Dict, api_key: str, base_url: str, timeout: float) -> Dict:
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    return json.loads(body)


def selected_prompts(args: argparse.Namespace) -> List[Dict]:
    rows = read_jsonl(Path(args.input_jsonl))
    pat = re.compile(args.id_regex) if args.id_regex else None
    out = []
    for rec in rows:
        cid = str(rec.get("custom_id") or rec.get("id") or "")
        if pat and not pat.search(cid):
            continue
        out.append(rec)
    if args.limit and args.limit > 0:
        out = out[: args.limit]
    return out


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("SCNET_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        raise SystemExit("[ERROR] SCNET_API_KEY is not set. Set it in the environment; do not put it in files.")

    out_path = Path(args.output_jsonl)
    failed_path = Path(args.failed_jsonl) if args.failed_jsonl else out_path.with_suffix(".failed.jsonl")
    prompts = selected_prompts(args)
    done = load_done_ids(out_path) if args.resume else set()
    pending = [p for p in prompts if str(p.get("custom_id") or p.get("id") or "") not in done]

    print(f"[INFO] selected prompts: {len(prompts)}")
    print(f"[INFO] already done: {len(done)}")
    print(f"[INFO] pending: {len(pending)}")
    print(f"[INFO] model: {args.model}")
    print(f"[INFO] base_url: {args.base_url}")
    if args.dry_run:
        for rec in pending[:3]:
            cid = str(rec.get("custom_id") or rec.get("id") or "")
            payload = make_payload(rec, args)
            print(json.dumps({"custom_id": cid, "payload_preview": payload}, ensure_ascii=False)[:2000])
        return

    for i, rec in enumerate(pending, 1):
        cid = str(rec.get("custom_id") or rec.get("id") or "")
        payload = make_payload(rec, args)
        last_err: Optional[str] = None
        last_response_preview = ""
        for attempt in range(1, args.max_retries + 1):
            try:
                response = call_scnet(payload, api_key, args.base_url, args.timeout)
                last_response_preview = json.dumps(response, ensure_ascii=False)[:3000]
                choice = (response.get("choices") or [{}])[0]
                content = response_message_text(choice)
                if not content.strip():
                    raise ValueError(f"empty model content; choice keys={sorted(choice.keys())}")
                parsed = parse_model_json(content)
                prior = coerce_prior(cid, parsed, content, response)
                append_jsonl(out_path, prior)
                print(f"[OK] {i}/{len(pending)} {cid}")
                last_err = None
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
                last_err = f"HTTP {e.code}: {body[:500]}"
            except Exception as e:
                last_err = repr(e)
            wait = args.sleep * (2 ** (attempt - 1)) + random.random() * 0.2
            print(f"[WARN] {cid} attempt {attempt}/{args.max_retries}: {last_err}; sleep {wait:.1f}s")
            time.sleep(wait)
        if last_err:
            append_jsonl(
                failed_path,
                {
                    "custom_id": cid,
                    "error": last_err,
                    "payload": payload,
                    "response_preview": last_response_preview,
                    "prompt_style": args.prompt_style,
                },
            )
            print(f"[FAIL] {cid}: {last_err}", file=sys.stderr)
        time.sleep(args.sleep)

    print("[DONE]")
    print("[OUT]", out_path)
    print("[FAILED]", failed_path)


if __name__ == "__main__":
    main()
