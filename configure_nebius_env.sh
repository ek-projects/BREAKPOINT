#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! grep -qxF '.env' .gitignore 2>/dev/null; then
  echo 'Refusing: .env is not listed in .gitignore' >&2
  exit 1
fi
printf 'Nebius API key: ' >&2
IFS= read -r -s KEY
printf '\n' >&2
if [[ -z "$KEY" ]]; then
  echo 'Empty key; nothing written.' >&2
  exit 1
fi
umask 077
{
  printf 'NEBIUS_API_KEY=%s\n' "$KEY"
  printf 'NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1\n'
  printf 'NEBIUS_MODEL=nvidia/Nemotron-3_5-Lightning\n'
} > .env
unset KEY
chmod 600 .env
printf 'Nebius credentials saved locally in %s/.env (mode 600).\n' "$PWD"
