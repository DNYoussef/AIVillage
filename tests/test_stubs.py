import pytest
from communications.credits import CreditLedger


def test_credit_ledger_simple():
    ledger = CreditLedger()
    ledger.credit("node1", 5)
    assert ledger.balance("node1") == 5
    ledger.debit("node1", 3)
    assert ledger.balance("node1") == 2
    with pytest.raises(ValueError):
        ledger.debit("node1", 5)
