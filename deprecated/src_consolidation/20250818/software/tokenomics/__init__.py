"""
Tokenomic Receipt System

Dual and triple layer system for:
- Clear communication tracking
- Action receipt generation
- Service payment handling
- Distributed compute compensation
- Agent interaction accounting
"""

from .credit_system import VILLAGECreditSystem
from .payment_processor import PaymentProcessor
from .receipt_system import ReceiptManager

__all__ = ["ReceiptManager", "VILLAGECreditSystem", "PaymentProcessor"]
