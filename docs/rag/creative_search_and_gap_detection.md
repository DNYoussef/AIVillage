# Creative Search and Gap Detection

This guide shows how to explore knowledge graphs creatively and how to detect missing nodes or connections. It complements the [Unified RAG Module Structure](../unified_rag_module_structure.md).

## CreativeGraphSearch patterns
`CreativeGraphSearch` supports multiple patterns of divergent thinking:
- **Analogical** – find analogies between distant concepts
- **Combinatorial** – combine unrelated concepts
- **Divergent** – explore multiple directions
- **Associative** – follow loose associations
- **Metaphorical** – create metaphorical connections
- **Contrarian** – explore contrasting ideas

### Example
```python
import networkx as nx
from unified_rag.graph import CreativeGraphSearch, CreativityPattern

G = nx.Graph()
search = CreativeGraphSearch(G)
session = await search.creative_brainstorm(
    "sustainable energy",
    creativity_patterns=[
        CreativityPattern.ANALOGICAL,
        CreativityPattern.COMBINATORIAL,
    ],
    num_insights=3,
)
for insight in session.generated_insights:
    print(insight.insight_text)
```

Query the system in creative mode via CLI:
```bash
python scripts/rag_cli.py query "How does photosynthesis relate to solar panels?" --mode creative
```

## MissingNodeDetector
`MissingNodeDetector` analyzes a graph and reports gaps such as missing concepts, missing connections, structural gaps and more. Typical output is a `GapAnalysis` with a `coverage_score`, recommendations and a list of `KnowledgeGap` entries containing gap type, description and suggested concepts or connections.

### Example
```python
import networkx as nx
from unified_rag.graph import MissingNodeDetector

G = nx.Graph()
detector = MissingNodeDetector(G)
analysis = await detector.detect_missing_nodes()
for gap in analysis.detected_gaps:
    print(gap.gap_type, gap.description)
```

Run gap detection from the command line:
```bash
python scripts/rag_cli.py detect-gaps
```
