from src.production.compression import test_pipeline


def test_verify_flag_triggers_message():
    assert test_pipeline.main(["--verify-4x-ratio"]) == "4x ratio verified"


def test_verify_flag_absent():
    assert test_pipeline.main([]) == "4x ratio not verified"
