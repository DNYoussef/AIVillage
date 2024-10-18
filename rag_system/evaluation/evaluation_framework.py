from typing import Dict, Any, List
import json
import time

class EvaluationFramework:
    """
    Framework to evaluate the performance of the system, measuring task classification accuracy and execution success rates.
    """

    def __init__(self):
        self.metrics = {
            'task_classification_accuracy': [],
            'execution_success_rate': [],
            'timestamps': []
        }

    def log_task_classification(self, predicted: str, actual: str):
        correct = int(predicted == actual)
        self.metrics['task_classification_accuracy'].append(correct)
        self.metrics['timestamps'].append(time.time())

    def log_execution_success(self, success: bool):
        self.metrics['execution_success_rate'].append(int(success))
        self.metrics['timestamps'].append(time.time())

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'task_classification_accuracy': self._calculate_accuracy(self.metrics['task_classification_accuracy']),
            'execution_success_rate': self._calculate_accuracy(self.metrics['execution_success_rate']),
            'total_tasks': len(self.metrics['task_classification_accuracy']),
            'timestamps': self.metrics['timestamps']
        }

    def _calculate_accuracy(self, results: List[int]) -> float:
        if not results:
            return 0.0
        return sum(results) / len(results)

    def save_metrics(self, file_path: str):
        with open(file_path, 'w') as f:
            json.dump(self.metrics, f)

    def load_metrics(self, file_path: str):
        with open(file_path, 'r') as f:
            self.metrics = json.load(f)

# Example usage
if __name__ == "__main__":
    evaluator = EvaluationFramework()
    evaluator.log_task_classification(predicted="general_query", actual="general_query")
    evaluator.log_execution_success(success=True)
    print(evaluator.get_metrics())
    evaluator.save_metrics("metrics.json")
