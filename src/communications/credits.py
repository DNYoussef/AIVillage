"""Simple in-memory credit ledger for mesh prototypes."""

from collections import defaultdict


class CreditLedger:
    def __init__(self):
        self._balances = defaultdict(int)

    def credit(self, node: str, amount: int):
        if amount < 0:
            raise ValueError("amount must be \u2265 0")
        self._balances[node] += amount

    def debit(self, node: str, amount: int):
        if amount < 0 or amount > self._balances[node]:
            raise ValueError("insufficient balance")
        self._balances[node] -= amount

    def balance(self, node: str) -> int:
        return self._balances[node]
