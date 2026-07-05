#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick connectivity test for SCNet/OpenAI-compatible LLM extraction endpoint."""
from __future__ import annotations

import argparse
import json
import os
from utils.llm_client import MultiEndpointClient, safe_json_extract


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--api-base', default=os.environ.get('API_BASES') or os.environ.get('SCNET_BASE_URL') or 'https://api.scnet.cn/api/llm/v1')
    ap.add_argument('--model-id', default=os.environ.get('MODEL_ID') or os.environ.get('SCNET_MODEL') or 'DeepSeek-R1-Distill-Qwen-32B')
    ap.add_argument('--timeout', type=int, default=120)
    args = ap.parse_args()
    llm = MultiEndpointClient([args.api_base], args.model_id, timeout=args.timeout, max_retries=1)
    prompt = 'Return JSON only: {"ok": true, "model_checked": "yes"}'
    txt = llm.completions(prompt, temperature=0.0, max_tokens=128)
    obj = safe_json_extract(txt)
    print(json.dumps({'api_base': args.api_base, 'model_id': args.model_id, 'raw': txt, 'json': obj}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
