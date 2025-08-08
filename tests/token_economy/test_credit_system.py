from pathlib import Path
import tempfile

from token_economy.credit_system import EarningRule, VILLAGECreditSystem


def test_earn_and_spend_credits():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "credits.db"
        cs = VILLAGECreditSystem(str(db_path))
        cs.add_earning_rule(EarningRule("LESSON", 10, {}, {}))
        earned = cs.earn_credits(
            "user1", "LESSON", {"location": "Nigeria", "country": "Nigeria"}
        )
        assert earned == int(10 * 1.5 * 1.3 * 2)
        balance = cs.get_balance("user1")
        assert balance == earned
        cs.spend_credits("user1", 5, "CONTENT", {})
        assert cs.get_balance("user1") == balance - 5
        cs.close()
