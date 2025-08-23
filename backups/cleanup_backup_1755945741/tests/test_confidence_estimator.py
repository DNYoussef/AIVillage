from rag_system.processing.confidence_estimator import ConfidenceEstimator


def test_estimate_confidence_updates_history():
    est = ConfidenceEstimator()
    ctx = [{"score": 0.8}, {"score": 0.6}]
    score = est.estimate_confidence("q", ctx, "resp")
    assert 0.0 <= score <= 1.0
    assert est.history[-1] == score


def test_update_model_appends_feedback():
    est = ConfidenceEstimator()
    est.update_model("q", [], "r", 0.5)
    assert est.history[-1] == 0.5
