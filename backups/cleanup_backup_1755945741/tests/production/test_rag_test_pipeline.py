from src.production.rag import test_pipeline


def test_query_performance_flag():
    result = test_pipeline.main(["--query-performance"])
    assert result == "performance evaluation complete"


def test_query_performance_default():
    result = test_pipeline.main([])
    assert result == "performance evaluation skipped"
