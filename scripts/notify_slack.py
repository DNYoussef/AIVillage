import os
import sys
import json
import requests

WEBHOOK = os.environ.get("SLACK_WEBHOOK")
if not WEBHOOK:
    sys.exit("no webhook")

msg = {
    "text": f"Atlantis CI passed for commit {os.environ.get('GITHUB_SHA','local')} :rocket:"
}
requests.post(WEBHOOK, data=json.dumps(msg))
