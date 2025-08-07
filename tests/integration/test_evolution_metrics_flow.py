from dataclasses import dataclass, field


@dataclass
class EvolutionMetrics:
    """Minimal metrics collector for tests."""

    rounds: int = 0
    scores: list[float] = field(default_factory=list)

    def record_round(self, score: float) -> None:
        self.rounds += 1
        self.scores.append(score)

    def best_score(self) -> float:
        return max(self.scores) if self.scores else 0.0


class FakePrometheus:
    """Very small Prometheus-style scraper used for tests."""

    def __init__(self) -> None:
        self.scrapes: list[dict[str, float]] = []

    def record(self, metrics: EvolutionMetrics) -> None:
        self.scrapes.append({"rounds": metrics.rounds, "best": metrics.best_score()})

    def scrape(self) -> dict[str, float]:
        return self.scrapes[-1] if self.scrapes else {}


def selector(metrics: EvolutionMetrics) -> float:
    """Example selector that picks the best score from metrics."""
    return metrics.best_score()


def run_evolution_round(metrics: EvolutionMetrics) -> None:
    """Simulate a single evolution round producing metrics."""
    # A deterministic "fitness" value to keep tests stable.
    metrics.record_round(score=0.92)


def test_evolution_metrics_flow() -> None:
    metrics = EvolutionMetrics()
    prom = FakePrometheus()

    # Trigger evolution and record metrics
    run_evolution_round(metrics)
    assert metrics.rounds == 1
    assert metrics.best_score() == 0.92

    # Simulate Prometheus scraping the metrics
    prom.record(metrics)
    scraped = prom.scrape()
    assert scraped["rounds"] == 1
    assert scraped["best"] == 0.92

    # Selector should use the real metrics
    assert selector(metrics) == 0.92
