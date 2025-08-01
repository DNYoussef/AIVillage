from typing import Any


class ConfidenceEstimator:
    def __init__(self):
        self.history: list[float] = []

    def estimate_confidence(
        self, query: str, context: list[dict[str, Any]], response: str
    ) -> float:
        """Estimate the confidence of a given response based on the query and context.

        :param query: The original query
        :param context: A list of context dictionaries used to generate the response
        :param response: The generated response
        :return: A confidence score between 0 and 1
        """
        scores = [ctx.get("score", 0.0) for ctx in context]
        score_mean = sum(scores) / len(scores) if scores else 0.0
        length_penalty = min(len(response) / 1000.0, 1.0)
        confidence = 0.6 * score_mean + 0.4 * length_penalty
        confidence = float(max(0.0, min(1.0, confidence)))
        self.history.append(confidence)
        return confidence

    def update_model(
        self,
        query: str,
        context: list[dict[str, Any]],
        response: str,
        human_feedback: float,
    ):
        """Update the confidence estimation model based on human feedback.

        :param query: The original query
        :param context: A list of context dictionaries used to generate the response
        :param response: The generated response
        :param human_feedback: Human-provided confidence score between 0 and 1
        """
        if 0.0 <= human_feedback <= 1.0:
            self.history.append(human_feedback)
