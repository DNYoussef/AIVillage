# rag_system/processing/confidence_estimator.py

def estimate_confidence(extrapolation: str) -> float:
    """
    Estimate the confidence of the extrapolation result.

    :param extrapolation: The extrapolated text from the LLM.
    :return: A confidence score between 0 and 1.
    """
    confidence_keywords = {
        "certain": 1.0,
        "likely": 0.8,
        "possible": 0.6,
        "unlikely": 0.3,
        "impossible": 0.0
    }
    
    lower_extrapolation = extrapolation.lower()
    for keyword, score in confidence_keywords.items():
        if keyword in lower_extrapolation:
            return score
    
    return 0.5  # Default confidence if no keywords are found
