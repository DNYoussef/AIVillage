#!/usr/bin/env python3
"""
Read the agent_test_message from the shared memory system and output results.
"""

import json
import sqlite3
import sys
from pathlib import Path


def read_agent_test_message():
    """Read the agent_test_message from the memory database."""
    db_path = "./.mcp/memory.db"

    if not Path(db_path).exists():
        print(f"[ERROR] Memory database not found at: {db_path}")
        return None

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT key, value, agent_type, timestamp, metadata
            FROM agent_messages
            WHERE key = ?
        """,
            ("agent_test_message",),
        )

        row = cursor.fetchone()

        if not row:
            print("[INFO] No message found with key: agent_test_message")
            print("[INFO] Checking all available messages...")

            # List all available keys
            cursor = conn.execute("SELECT key FROM agent_messages")
            all_keys = cursor.fetchall()
            if all_keys:
                print(f"[INFO] Available message keys: {[k[0] for k in all_keys]}")
            else:
                print("[INFO] No messages found in database")
            return None

        key, value, agent_type, timestamp, metadata = row

        print("=" * 60)
        print("AGENT TEST MESSAGE FOUND:")
        print("=" * 60)
        print(f"Key: {key}")
        print(f"Value: {value}")
        print(f"Agent Type: {agent_type}")
        print(f"Timestamp: {timestamp}")

        try:
            meta = json.loads(metadata)
            print("Metadata:")
            for k, v in meta.items():
                print(f"  {k}: {v}")
        except BaseException:
            print(f"Metadata (raw): {metadata}")

        print("=" * 60)

        return {
            "key": key,
            "value": value,
            "agent_type": agent_type,
            "timestamp": timestamp,
            "metadata": metadata,
        }

    except Exception as e:
        print(f"[ERROR] Database error: {e}")
        return None
    finally:
        conn.close()


if __name__ == "__main__":
    result = read_agent_test_message()
    if result:
        sys.exit(0)
    else:
        sys.exit(1)
