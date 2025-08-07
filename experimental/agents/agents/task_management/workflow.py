from dataclasses import dataclass, field

from .task import Task


@dataclass(frozen=True)
class Workflow:
    id: str
    name: str
    tasks: list[Task]
    dependencies: dict[str, list[str]] = field(default_factory=dict)

    def add_task(self, task: Task) -> "Workflow":
        return Workflow(
            id=self.id,
            name=self.name,
            tasks=self.tasks + [task],
            dependencies=self.dependencies,
        )

    def update_task(self, updated_task: Task) -> "Workflow":
        new_tasks = [
            updated_task if task.id == updated_task.id else task for task in self.tasks
        ]
        return Workflow(
            id=self.id, name=self.name, tasks=new_tasks, dependencies=self.dependencies
        )

    def add_dependency(self, task_id: str, dependency_id: str) -> "Workflow":
        new_dependencies = self.dependencies.copy()
        if task_id not in new_dependencies:
            new_dependencies[task_id] = []
        new_dependencies[task_id].append(dependency_id)
        return Workflow(
            id=self.id, name=self.name, tasks=self.tasks, dependencies=new_dependencies
        )
