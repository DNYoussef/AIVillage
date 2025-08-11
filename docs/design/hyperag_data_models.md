# HypeRAG Data Models

The MCP server manages two memory structures and several auxiliary records. The YAML below summarises the key fields.

```yaml
Hyperedge:
  id: string
  entities: string[]        # n-ary
  relation: string
  confidence: float         # Bayesian
  timestamp: datetime
  tags: string[]            # ["creative","repaired",…]
  popularity_rank: int
  alpha_weight: float|null  # Rel-GAT
  source_docs: string[]
  embedding: vector

HippoNode:
  id: string
  content: string
  user_id: string|null      # Digital Twin
  episodic: bool
  created: datetime
  novelty_score_cache: float|null
  gdc_flags: string[]
  embedding: vector

GDCSpec:
  id: string
  description: string
  cypher: string
  severity: string          # low|med|high
  suggested_action: string
```
