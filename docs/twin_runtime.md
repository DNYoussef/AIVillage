# Twin Runtime

The digital twin runtime exposes a small `chat` interface which streams tokens
from a local language model.  Each prompt passes through `risk_gate` before any
tokens are generated.  The guard consults encrypted preferences stored in
`~/.aivillage/prefs.json.enc` and will:

- deny filesystem, process or network commands unless `allow_shell` is enabled
  in the preference file;
- deny messages that appear to exfiltrate secrets; and
- require user confirmation (`ask`) when an unknown tool is requested.

This minimal loop is enough for tests and demonstrations while remaining fully
offline by default.
