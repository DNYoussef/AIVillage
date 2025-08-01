# Agent Forge Security Model

This document outlines the basic security precautions used when executing
user supplied code in the ADAS subsystem.

## ADAS technique execution

`AgentTechnique.handle` runs small code snippets provided by the user. To
reduce the impact of malicious code the following measures are taken:

- Only a minimal set of builtin functions are exposed: `__import__`, `open`,
  `len`, `range`, `min`, `max`, `str` and `float`.
- The code is parsed with the Python `ast` module and any import statement is
  checked against a small allow list (currently only `os`). Calls to dangerous
  functions such as `exec`, `eval` or `compile` are rejected.
- The code is compiled before execution so that syntax errors are caught
  safely.
- Each run happens inside a fresh temporary directory to avoid unwanted
  file system interactions.

These checks are intentionally simple and should not be considered a
complete sandbox. They merely reduce the attack surface for running
untrusted techniques.
