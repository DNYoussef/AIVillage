import tempfile
from pathlib import Path

from token_economy.compute_mining import ComputeMiningSystem, ComputeSession
from token_economy.credit_system import EarningRule, VILLAGECreditSystem


def test_compute_mining_awards_credits():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "credits.db"
        cs = VILLAGECreditSystem(str(db_path))
        cs.add_earning_rule(EarningRule("COMPUTE_CONTRIBUTION", 10, {}, {}))
        miner = ComputeMiningSystem(cs)
        session = ComputeSession(
            user_id="user1",
            operations=1000,
            duration=60,
            model_id="modelA",
            proof="valid",
            used_green_energy=True,
            device_location="Sub-Saharan Africa",
        )
        earned = miner.track_compute_contribution("device1", session)
        assert earned > 0
        assert cs.get_balance("user1") == earned
