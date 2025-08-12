from scripts.agent_kpi_system import AgentKPITracker, KPIMetric


def test_specialization_weights_differ() -> None:
    king = AgentKPITracker("a1", "king")
    magi = AgentKPITracker("a2", "magi")
    assert (
        king.kpi_weights[KPIMetric.INTER_AGENT_COOPERATION]
        != magi.kpi_weights[KPIMetric.INTER_AGENT_COOPERATION]
    )
    assert (
        magi.kpi_weights[KPIMetric.OUTPUT_QUALITY_SCORE]
        > king.kpi_weights[KPIMetric.OUTPUT_QUALITY_SCORE]
    )
