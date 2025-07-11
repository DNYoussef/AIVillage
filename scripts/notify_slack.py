import os
import json
import requests
import subprocess
import re

WEBHOOK = os.getenv("SLACK_WEBHOOK")
CHANNEL = os.getenv("SLACK_CHANNEL", "#atlantis-dev")
if not WEBHOOK:
    raise SystemExit("$SLACK_WEBHOOK not set")

# Grab commit info
_git = lambda *args: subprocess.check_output(["git", *args], text=True).strip()
sha = _git("rev-parse", "--short", "HEAD")
msg = _git("log", "-1", "--pretty=%s")
author = _git("log", "-1", "--pretty=%an")

# Extract ADR links touched in commit diff
files = _git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").split()
adr_links = [f for f in files if re.match(r"docs/adr/\d+.*\.md", f)]
if adr_links:
    adr_section = "\n> _Updated ADRs:_\n" + "\n".join(
        f"\u2022 <https://github.com/AtlantisAI/atlantis/blob/main/{p}|{p}>"
        for p in adr_links
    )
else:
    adr_section = ""

payload = {
    "channel": CHANNEL,
    "text": f"✅ *CI green* for commit `{sha}` by *{author}* — {msg}{adr_section}",
}
requests.post(WEBHOOK, data=json.dumps(payload), timeout=5)
