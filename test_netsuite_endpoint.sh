#!/usr/bin/env bash
set -e

BASE_URL="${BASE_URL:-https://netsuite-ai-production.up.railway.app}"

echo "Testing /chat ..."
curl -s "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"How do I create an invoice in NetSuite?"}'
echo
echo
echo "Testing /v1/chat/completions ..."
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role":"user","content":"How do I accept customer payments?"}
    ]
  }'
echo
