"""Offline Amazon order history parser.

Amazon allows users to export their order history as a CSV file.  This
connector reads such an export and exposes the parsed orders without making
any network requests.  Only a tiny subset of columns is required for the
parser: ``Order Date``, ``Title`` and ``Total Owed`` (or ``Total Charged``).
Additional columns are ignored but preserved in the returned dictionaries.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Dict


class AmazonOrdersConnector:
    """Parser for Amazon order history CSV exports."""

    def __init__(self, csv_path: str | None = None) -> None:
        self._orders: List[Dict[str, str]] = []
        if csv_path:
            self.load_export(csv_path)

    def load_export(self, csv_path: str | Path) -> int:
        """Load orders from a CSV export file.

        The number of orders parsed is returned.
        """
        self._orders.clear()
        path = Path(csv_path)
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not any(row.values()):
                    continue
                order = dict(row)  # keep all columns for callers
                self._orders.append(order)
        return len(self._orders)

    def get_orders(self) -> List[Dict[str, str]]:
        return list(self._orders)

    def get_order_count(self) -> int:
        return len(self._orders)


# Backwards compatibility helpers -------------------------------------------

def get_orders(csv_path: str | Path) -> List[Dict[str, str]]:
    return AmazonOrdersConnector(csv_path).get_orders()


def get_order_count(csv_path: str | Path) -> int:
    return AmazonOrdersConnector(csv_path).get_order_count()
