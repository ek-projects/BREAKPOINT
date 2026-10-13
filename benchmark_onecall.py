from __future__ import annotations
import json
import os
import time
from pathlib import Path

# Load local .env without printing secret values.
for line in Path('.env').read_text().splitlines():
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    key, value = line.split('=', 1)
    os.environ.setdefault(key.strip(), value.strip())
os.environ['NEBIUS_MAX_TOKENS'] = os.getenv('TEST_MAX_TOKENS', '192')
os.environ['NEBIUS_ENABLE_THINKING'] = 'false'

from demo_runtime import run_demo
from nebius_reasoner import NemotronClient

for mode in ('deterministic', 'live'):
    print('MODE', mode)
    for fault in ('database', 'queue', 'dependency'):
        client = None if mode == 'deterministic' else NemotronClient()
        started = time.perf_counter()
        result = run_demo(fault, client=client, max_architect_calls=1)
        elapsed = time.perf_counter() - started
        safe = {
            'truth': fault,
            'prediction': result['prediction'],
            'correct': result['correct'],
            'confidence': round(result['confidence'], 6),
            'steps': len(result['steps']),
            'architect_attempts': result['architect_attempts'],
            'architect_calls': result['architect_calls'],
            'nemotron_invoked': [s['nemotron_invoked'] for s in result['steps']],
            'fallback': [s['fallback_used'] for s in result['steps']],
            'latency_s': round(elapsed, 3),
            'model_call_latency_s': [s['nemotron_metrics'].get('latency_s') for s in result['steps'] if s['nemotron_metrics']],
            'usage': [s['nemotron_metrics'].get('usage') for s in result['steps'] if s['nemotron_metrics']],
        }
        print(json.dumps(safe, separators=(',', ':')))
