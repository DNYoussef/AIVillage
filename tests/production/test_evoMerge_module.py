from src.production.evolution import test_evoMerge


def test_fitness_check_flag():
    assert test_evoMerge.main(["--fitness-check"]) is True


def test_fitness_check_default_false():
    assert test_evoMerge.main([]) is False
