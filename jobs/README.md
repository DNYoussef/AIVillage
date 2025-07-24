# HypeRAG Jobs

Automated batch jobs for HypeRAG knowledge graph maintenance and discovery.

## Hidden Link Scanner

**File:** `hyperag_scan_hidden_links.py`

Nightly cron job that surfaces candidate missing edges through co-mention analysis and divergent retrieval.

### Pipeline Steps

1. **Hippo-Index Log Analysis** - Query logs for high co-mention entity pairs
2. **Divergent Retrieval** - Call DivergentRetriever in scan mode (n=3 candidates per pair)
3. **Innovator → Guardian Pipeline** - Process candidates through validation
4. **Metrics Collection** - Track candidates found, guardian decisions, and performance

### Configuration

**Schedule:** `02:30 UTC` (configured in `config/scanner_config.json`)

**Key Settings:**
- `lookback_hours: 24` - Hours of log history to analyze
- `min_co_mentions: 3` - Minimum co-mentions to consider a pair
- `max_pairs_to_scan: 50` - Limit for performance
- `candidates_per_pair: 3` - DivergentRetriever scan depth

### Usage

```bash
# Run manually with default config
python jobs/hyperag_scan_hidden_links.py

# Run with custom config
python jobs/hyperag_scan_hidden_links.py --config config/scanner_config.json

# Dry run (no changes applied)
python jobs/hyperag_scan_hidden_links.py --dry-run

# Verbose logging
python jobs/hyperag_scan_hidden_links.py --verbose
```

### Installation

1. **Install cron schedule:**
   ```bash
   # Edit paths in crontab_hyperag first
   crontab jobs/crontab_hyperag
   ```

2. **Verify installation:**
   ```bash
   crontab -l
   ```

3. **Test components:**
   ```bash
   python jobs/test_scanner.py
   ```

### Output

**Metrics:** Written to `data/scan_metrics/[scan_id]_metrics.json`

**Logs:** Written to `data/scan_logs/hidden_link_scan_[date].log`

**Example metrics:**
```json
{
  "scan_id": "scan_20250723_023000",
  "total_time_seconds": 45.2,
  "input_metrics": {
    "co_mention_pairs_found": 127,
    "high_confidence_pairs": 23
  },
  "pipeline_metrics": {
    "guardian_approved": 12,
    "guardian_quarantined": 8,
    "guardian_rejected": 3
  }
}
```

## Cron Jobs

**Schedule Overview:**
- **02:30 UTC Daily** - Hidden link scanner
- **03:00 UTC Sundays** - Log cleanup
- **04:00 UTC Monthly** - Metrics aggregation

**Files:**
- `crontab_hyperag` - Cron configuration
- `hyperag_scan_hidden_links.py` - Main scanner
- `test_scanner.py` - Test suite

## Dependencies

- HypeRAG modules: `mcp_servers.hyperag.*`
- Python 3.8+
- Required for production: Hippo-Index logs, DivergentRetriever, Guardian Gate

## Performance

**Target metrics:**
- Process 50 entity pairs in <60 seconds
- Guardian evaluation <20ms per candidate
- Handle 24 hours of Hippo-Index logs efficiently

**Monitoring:**
- Error rate alerts if >10% failures
- Performance alerts if >90 seconds total time
- Audit all Guardian decisions for review
