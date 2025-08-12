from token_economy.jurisdiction import (
    JurisdictionManager,
    JurisdictionType,
    UserContext,
)


def test_jurisdiction_detection_and_rules():
    jm = JurisdictionManager()
    ctx = UserContext(
        gps_location="China", ip_address=None, sim_country=None, device_locale=None
    )
    jt = jm.detect_jurisdiction(ctx)
    assert jt == JurisdictionType.RED
    jm.apply_jurisdiction_rules("user1", jt)
    assert jm.user_modes["user1"] == "EDUCATION_ONLY"
    assert "crypto" in jm.disabled_features["user1"]

    ctx2 = UserContext(
        gps_location="Kenya", ip_address=None, sim_country=None, device_locale=None
    )
    jt2 = jm.detect_jurisdiction(ctx2)
    assert jt2 == JurisdictionType.GREEN
