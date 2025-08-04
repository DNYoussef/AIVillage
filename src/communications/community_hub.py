import logging
from typing import Any

from core.error_handling import StandardCommunicationProtocol

logger = logging.getLogger(__name__)


class CommunityHub:
    def __init__(self, communication_protocol: StandardCommunicationProtocol) -> None:
        self.research_results = {}
        self.agents = {}
        self.projects = {}
        self.communication_protocol = communication_protocol

    async def get_all_projects(self) -> dict[str, Any]:
        return self.projects

    async def assign_task(self, task_id: str, agent_id: str) -> None:
        if agent_id not in self.agents:
            msg = f"Agent {agent_id} not found"
            raise ValueError(msg)
        self.agents[agent_id]["tasks"].append(task_id)
        logger.info(f"Task {task_id} assigned to agent {agent_id}")

    async def reassign_task(self, task_id: str, new_agent_id: str) -> None:
        for agent_data in self.agents.values():
            if task_id in agent_data["tasks"]:
                agent_data["tasks"].remove(task_id)
                break
        await self.assign_task(task_id, new_agent_id)
        logger.info(f"Task {task_id} reassigned to agent {new_agent_id}")

    async def add_resources_to_project(self, project_id: str, resources: dict[str, Any]) -> None:
        if project_id not in self.projects:
            msg = f"No project found with ID {project_id}"
            raise ValueError(msg)
        self.projects[project_id].setdefault("resources", {}).update(resources)
        logger.info(f"Resources added to project {project_id}")

    async def update_project_data(self, task_id: str, data: dict[str, Any]) -> None:
        for project_id, project_data in self.projects.items():
            if task_id in project_data.get("tasks", []):
                project_data.setdefault("task_data", {})[task_id] = data
                logger.info(f"Updated data for task {task_id} in project {project_id}")
                break

    async def request_collaboration(self, requester_id: str, task_id: str, required_capabilities: list[str]) -> str:
        for agent_id, agent_data in self.agents.items():
            if agent_id != requester_id and all(cap in agent_data["capabilities"] for cap in required_capabilities):
                await self.assign_task(task_id, agent_id)
                logger.info(f"Collaboration request from {requester_id} for task {task_id} assigned to {agent_id}")
                return agent_id
        logger.warning(f"No suitable agent found for collaboration request from {requester_id} for task {task_id}")
        return ""

    async def post_research_results(self, task_id: str, results: dict[str, Any]) -> None:
        self.research_results[task_id] = results
        logger.info(f"Research results for task {task_id} posted")

    async def get_research_results(self, task_id: str) -> dict[str, Any]:
        return self.research_results.get(task_id, {})

    async def update_project_status(self, project_id: str, status: str, progress: float) -> None:
        if project_id not in self.projects:
            msg = f"No project found with ID {project_id}"
            raise ValueError(msg)
        self.projects[project_id]["status"] = status
        self.projects[project_id]["progress"] = progress
        logger.info(f"Updated status of project {project_id} to {status} with progress {progress}")

    async def generate_project_report(self, project_id: str):
        """Generate a summary report for a single project.

        The report aggregates stored project information such as tasks,
        resources and any research results for those tasks.
        """
        if project_id not in self.projects:
            msg = f"No project found with ID {project_id}"
            raise ValueError(msg)

        project_data = self.projects[project_id]
        task_reports = []
        for task_id in project_data.get("tasks", []):
            task_reports.append(
                {
                    "task_id": task_id,
                    "data": project_data.get("task_data", {}).get(task_id, {}),
                    "research_results": self.research_results.get(task_id, {}),
                }
            )

        report = {
            "project_id": project_id,
            "status": project_data.get("status"),
            "progress": project_data.get("progress"),
            "resources": project_data.get("resources", {}),
            "tasks": task_reports,
        }

        logger.info(f"Generated project report for {project_id}")
        return report

    async def create_combined_report(self, project_ids: list[str]) -> dict[str, Any]:
        """Create a combined report for multiple projects."""
        combined_report = {"projects": [], "overall_progress": 0.0}

        progress_sum = 0.0
        for pid in project_ids:
            report = await self.generate_project_report(pid)
            combined_report["projects"].append(report)
            progress_sum += report.get("progress", 0.0) or 0.0

        if project_ids:
            combined_report["overall_progress"] = progress_sum / len(project_ids)

        logger.info(f"Created combined report for projects: {', '.join(project_ids)}")
        return combined_report
