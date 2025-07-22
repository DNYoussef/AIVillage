# HypeRAG MCP API

The MCP server exposes REST-style endpoints for agent access. Authentication is role-based and enforced by the MCP core and Guardian Gate.

| Route                           | Method | Auth  | Description                                                  |
| ------------------------------- | ------ | ----- | ------------------------------------------------------------ |
| `/v1/hyperag/query`             | POST   | read  | Normal/creative/repair queries (`mode`, `plan_hints`).       |
| `/v1/hyperag/creative`          | POST   | read  | Divergent bridge search between `source` & `target`.         |
| `/v1/hyperag/repair`            | POST   | write | Submit GDC violation id or subgraph for Innovator proposals. |
| `/v1/hyperag/guardian/validate` | POST   | write | Manually vet proposal set.                                   |
| `/v1/hyperag/adapter/upload`    | POST   | write | Upload LoRA; Guardian signs.                                 |

Every response returns `confidence`, `guardian_decision` and `reasoning_path` fields.

## Permission Roles
- **king, sage** – read/write
- **magi, watcher, external** – read-only
- **guardian** – override gate decisions
- **innovator** – propose only
