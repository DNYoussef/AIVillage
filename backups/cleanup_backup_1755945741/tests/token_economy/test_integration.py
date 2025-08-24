from pathlib import Path
import tempfile

from token_economy.compute_mining import ComputeMiningSystem, ComputeSession
from token_economy.credit_system import EarningRule, VILLAGECreditSystem
from token_economy.jurisdiction import JurisdictionManager, JurisdictionType, UserContext


def test_end_to_end_flow():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "credits.db"
        cs = VILLAGECreditSystem(str(db_path))
        cs.add_earning_rule(EarningRule("COMPUTE_CONTRIBUTION", 10, {}, {}))
        jm = JurisdictionManager()
        ctx = UserContext(
            gps_location="Nigeria",
            ip_address=None,
            sim_country=None,
            device_locale=None,
        )
        jt = jm.detect_jurisdiction(ctx)
        assert jt == JurisdictionType.YELLOW
        miner = ComputeMiningSystem(cs)
        session = ComputeSession(
            user_id="user1",
            operations=500,
            duration=30,
            model_id="modelB",
            proof="valid",
            used_green_energy=False,
            device_location="Sub-Saharan Africa",
        )
        miner.track_compute_contribution("device1", session)
        assert cs.get_balance("user1") > 0
