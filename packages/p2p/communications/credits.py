"""Simple in-memory credit ledger for mesh prototypes."""

from collections import defaultdict


class CreditLedger:
    def __init__(self) -> None:
        self._balances = defaultdict(int)

    def credit(self, node: str, amount: int) -> None:
        if amount < 0:
            msg = "amount must be \u2265 0"
            raise ValueError(msg)
        self._balances[node] += amount

    def debit(self, node: str, amount: int) -> None:
        if amount < 0 or amount > self._balances[node]:
            msg = "insufficient balance"
            raise ValueError(msg)
        self._balances[node] -= amount

    def balance(self, node: str) -> int:
        return self._balances[node]
