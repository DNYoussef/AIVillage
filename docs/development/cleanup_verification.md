# Cleanup Verification

The project includes a script to verify cleanup actions and record metrics for future audits.

## Usage

1. Generate a log file where each line describes a cleanup action.
2. Run the verification script:
   ```bash
   python tools/verify_cleanup.py path/to/cleanup.log
   ```
3. The script logs each action, counts them, and writes the total to `cleanup_metrics.json`.

The `cleanup_metrics.json` file is kept in the repository to make cleanup claims reproducible.
