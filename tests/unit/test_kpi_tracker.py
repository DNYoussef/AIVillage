from datetime import timedelta

from scripts.agent_kpi_system import AgentKPITracker, KPIMetric


def test_kpi_tracking_records_and_scores() -> None:
    tracker = AgentKPITracker("agent1", "king")
    tracker.record_performance("task1", {KPIMetric.TASK_COMPLETION_RATE: 0.8})
    tracker.record_performance("task2", {KPIMetric.TASK_COMPLETION_RATE: 0.9})
    score = tracker.calculate_overall_kpi()
    assert 0 < score <= 1
    recent = tracker.calculate_overall_kpi(time_window=timedelta(days=1))
    assert recent > 0
