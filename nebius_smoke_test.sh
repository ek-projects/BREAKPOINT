#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f .env ]]; then
  echo 'Missing .env. Run ./configure_nebius_env.sh first.' >&2
  exit 2
fi
set -a
# shellcheck disable=SC1091
. ./.env
set +a
: "${NEBIUS_API_KEY:?missing NEBIUS_API_KEY}"
: "${NEBIUS_BASE_URL:=https://api.tokenfactory.nebius.com/v1}"
: "${NEBIUS_MODEL:=nvidia/Nemotron-3_5-Lightning}"
python3 - <<'PY'
import json, os, urllib.request, urllib.error, time
base=os.environ['NEBIUS_BASE_URL'].rstrip('/')
key=os.environ['NEBIUS_API_KEY']
model=os.environ['NEBIUS_MODEL']
url=base+'/chat/completions'
payload=json.dumps({
  'model': model,
  'messages': [{'role':'user','content':'Reply with exactly: PONG'}],
  'max_tokens': 16,
  'temperature': 0
}).encode()
req=urllib.request.Request(url, data=payload, headers={
  'Authorization':'Bearer '+key,
  'Content-Type':'application/json'
}, method='POST')
t0=time.time()
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        raw=r.read().decode('utf-8','replace')
        elapsed=time.time()-t0
        data=json.loads(raw)
        text=((data.get('choices') or [{}])[0].get('message') or {}).get('content','')
        usage=data.get('usage') or {}
        print(json.dumps({
          'ok': True,
          'http_status': r.status,
          'model': data.get('model', model),
          'reply': text,
          'latency_s': round(elapsed,3),
          'usage': usage
        }, ensure_ascii=False))
except urllib.error.HTTPError as e:
    body=e.read().decode('utf-8','replace')
    print(json.dumps({'ok':False,'http_status':e.code,'error_body':body[:1200]}, ensure_ascii=False))
    raise SystemExit(1)
PY
