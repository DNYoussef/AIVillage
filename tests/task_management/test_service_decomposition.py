"""
Comprehensive test suite for the decomposed task management services.
Tests each service independently and the facade integration.
"""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

from experiments.agents.agents.task_management.services.task_creation_service import TaskCreationService
from experiments.agents.agents.task_management.services.task_assignment_service import TaskAssignmentService
from experiments.agents.agents.task_management.services.task_execution_service import TaskExecutionService
from experiments.agents.agents.task_management.services.task_completion_service import TaskCompletionService
from experiments.agents.agents.task_management.services.project_management_service import ProjectManagementService
from experiments.agents.agents.task_management.services.incentive_service import IncentiveService
from experiments.agents.agents.task_management.services.analytics_service import AnalyticsService
from experiments.agents.agents.task_management.services.persistence_service import PersistenceService
from experiments.agents.agents.task_management.services.unified_task_manager_facade import UnifiedTaskManagerFacade

from experiments.agents.agents.task_management.task import Task, TaskStatus
from experiments.agents.agents.task_management.unified_task_manager import Project


class TestTaskCreationService:
    """Test TaskCreationService functionality."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        subgoal_generator = Mock()
        assignment_service = Mock()
        project_service = Mock()
        return subgoal_generator, assignment_service, project_service
    
    @pytest.fixture
    def service(self, mock_dependencies):
        """Create TaskCreationService instance."""
        return TaskCreationService(*mock_dependencies)
    
    @pytest.mark.asyncio
    async def test_create_task_success(self, service):
        """Test successful task creation."""
        task = await service.create_task(
            description="Test task",
            agent="test_agent",
            priority=1
        )
        
        assert task.description == "Test task"
        assert task.assigned_agents == ["test_agent"]
        assert task.priority == 1
        assert task.id is not None
    
    @pytest.mark.asyncio
    async def test_create_complex_task(self, service, mock_dependencies):
        """Test complex task creation with subgoals."""
        subgoal_generator, assignment_service, project_service = mock_dependencies
        
        # Setup mocks
        subgoal_generator.generate_subgoals = AsyncMock(return_value=["subgoal1", "subgoal2"])
        assignment_service.select_best_agent_for_task = AsyncMock(return_value="agent1")
        
        tasks = await service.create_complex_task("Complex task", {"context": "test"})
        
        assert len(tasks) == 2
        assert all(isinstance(task, Task) for task in tasks)
    
    def test_validate_task_data(self, service):
        """Test task data validation."""
        assert service.validate_task_data("Valid task", "valid_agent") is True
        assert service.validate_task_data("", "valid_agent") is False
        assert service.validate_task_data("Valid task", "") is False


class TestTaskAssignmentService:
    """Test TaskAssignmentService functionality."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        communication_protocol = Mock()
        decision_maker = Mock()
        incentive_service = Mock()
        return communication_protocol, decision_maker, incentive_service
    
    @pytest.fixture
    def service(self, mock_dependencies):
        """Create TaskAssignmentService instance."""
        return TaskAssignmentService(*mock_dependencies)
    
    @pytest.mark.asyncio
    async def test_assign_task(self, service, mock_dependencies):
        """Test task assignment."""
        communication_protocol, decision_maker, incentive_service = mock_dependencies
        
        # Setup mocks
        incentive_service.calculate_incentive = AsyncMock(return_value=10.0)
        communication_protocol.send_message = AsyncMock()
        
        task = Task(description="Test task", assigned_agents=["agent1"])
        await service.assign_task(task)
        
        # Verify task was moved to ongoing
        ongoing_tasks = service.get_ongoing_tasks()
        assert task.id in ongoing_tasks
        assert ongoing_tasks[task.id].status == TaskStatus.IN_PROGRESS
    
    @pytest.mark.asyncio
    async def test_select_best_agent(self, service, mock_dependencies):
        """Test agent selection."""
        communication_protocol, decision_maker, incentive_service = mock_dependencies
        
        # Setup mocks
        decision_maker.make_decision = AsyncMock(return_value={"best_alternative": "best_agent"})
        service.update_agent_list(["agent1", "agent2", "best_agent"])
        
        selected_agent = await service.select_best_agent_for_task("Test task")
        assert selected_agent == "best_agent"


class TestTaskExecutionService:
    """Test TaskExecutionService functionality."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        communication_protocol = Mock()
        creation_service = Mock()
        completion_service = Mock()
        return communication_protocol, creation_service, completion_service
    
    @pytest.fixture
    def service(self, mock_dependencies):
        """Create TaskExecutionService instance."""
        return TaskExecutionService(*mock_dependencies)
    
    @pytest.mark.asyncio
    async def test_process_single_task(self, service, mock_dependencies):
        """Test single task processing."""
        communication_protocol, creation_service, completion_service = mock_dependencies
        
        # Setup mocks
        communication_protocol.send_and_wait = AsyncMock(return_value={"result": "success"})
        
        task = Task(description="Test task", assigned_agents=["agent1"])
        result = await service.process_single_task(task)
        
        assert result == {"result": "success"}
    
    def test_batch_size_management(self, service):
        """Test batch size configuration."""
        service.set_batch_size(10)
        assert service.get_batch_size() == 10
        
        with pytest.raises(Exception):  # Should raise on invalid size
            service.set_batch_size(-1)


class TestTaskCompletionService:
    """Test TaskCompletionService functionality."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        assignment_service = Mock()
        incentive_service = Mock()
        analytics_service = Mock()
        project_service = Mock()
        return assignment_service, incentive_service, analytics_service, project_service
    
    @pytest.fixture
    def service(self, mock_dependencies):
        """Create TaskCompletionService instance."""
        return TaskCompletionService(*mock_dependencies)
    
    @pytest.mark.asyncio
    async def test_complete_task(self, service, mock_dependencies):
        """Test task completion."""
        assignment_service, incentive_service, analytics_service, project_service = mock_dependencies
        
        # Setup mocks
        task = Task(description="Test task", assigned_agents=["agent1"])
        assignment_service.get_ongoing_tasks.return_value = {"task_id": task}
        assignment_service.remove_ongoing_task.return_value = task
        incentive_service.update_agent_performance = AsyncMock()
        analytics_service.record_task_completion = AsyncMock()
        
        await service.complete_task("task_id", {"success": True})
        
        # Verify task was completed
        completed_tasks = service.get_completed_tasks()
        assert len(completed_tasks) == 1
        assert completed_tasks[0].status == TaskStatus.COMPLETED


class TestProjectManagementService:
    """Test ProjectManagementService functionality."""
    
    @pytest.fixture
    def service(self):
        """Create ProjectManagementService instance."""
        return ProjectManagementService()
    
    @pytest.mark.asyncio
    async def test_create_project(self, service):
        """Test project creation."""
        project_id = await service.create_project("Test Project", "Test Description")
        
        assert project_id is not None
        project = await service.get_project(project_id)
        assert project.name == "Test Project"
        assert project.description == "Test Description"
    
    @pytest.mark.asyncio
    async def test_update_project_status(self, service):
        """Test project status updates."""
        project_id = await service.create_project("Test Project", "Test Description")
        
        await service.update_project_status(project_id, status="in_progress", progress=0.5)
        
        project = await service.get_project(project_id)
        assert project.status == "in_progress"
        assert project.progress == 0.5
    
    @pytest.mark.asyncio
    async def test_add_task_to_project(self, service):
        """Test adding tasks to projects."""
        project_id = await service.create_project("Test Project", "Test Description")
        
        await service.add_task_to_project(
            project_id, 
            "task_1", 
            {"description": "Test task", "agent": "agent1"}
        )
        
        tasks = await service.get_project_tasks(project_id)
        assert len(tasks) == 1
        assert tasks[0].description == "Test task"


class TestIncentiveService:
    """Test IncentiveService functionality."""
    
    @pytest.fixture
    def service(self):
        """Create IncentiveService instance."""
        incentive_model = Mock()
        unified_analytics = Mock()
        return IncentiveService(incentive_model, unified_analytics)
    
    @pytest.mark.asyncio
    async def test_calculate_incentive(self, service):
        """Test incentive calculation."""
        service._incentive_model.calculate_incentive.return_value = {"incentive": 15.0}
        
        task = Task(description="Test task")
        incentive = await service.calculate_incentive("agent1", task)
        
        assert incentive == 15.0
    
    def test_agent_performance_tracking(self, service):
        """Test agent performance tracking."""
        # Test getting performance for non-existent agent
        assert service.get_agent_performance("agent1") == 0.0
        
        # Test getting top performers when empty
        assert service.get_top_performing_agents() == []


class TestAnalyticsService:
    """Test AnalyticsService functionality."""
    
    @pytest.fixture
    def service(self):
        """Create AnalyticsService instance."""
        unified_analytics = Mock()
        unified_analytics.generate_summary_report.return_value = {"test": "report"}
        return AnalyticsService(unified_analytics)
    
    @pytest.mark.asyncio
    async def test_record_task_completion(self, service):
        """Test task completion recording."""
        await service.record_task_completion("task_1", 10.5, True)
        
        metrics = await service.get_task_metrics("task_1")
        assert metrics is not None
        assert metrics["success"] is True
        assert metrics["completion_time"] == 10.5
    
    @pytest.mark.asyncio
    async def test_generate_performance_report(self, service):
        """Test performance report generation."""
        await service.record_task_completion("task_1", 10.0, True)
        await service.record_task_completion("task_2", 20.0, False)
        
        report = await service.generate_performance_report()
        
        assert "unified_analytics" in report
        assert "task_metrics" in report
        assert report["task_metrics"]["total_tasks"] == 2
        assert report["task_metrics"]["success_rate"] == 0.5


class TestPersistenceService:
    """Test PersistenceService functionality."""
    
    @pytest.fixture
    def service(self, tmp_path):
        """Create PersistenceService instance."""
        service = PersistenceService()
        service._backup_directory = tmp_path / "backups"
        service._backup_directory.mkdir()
        return service
    
    @pytest.mark.asyncio
    async def test_save_and_load_state(self, service, tmp_path):
        """Test state save and load functionality."""
        test_file = tmp_path / "test_state.json"
        test_data = {
            "tasks": [{"id": "task_1", "description": "Test task"}],
            "projects": {"proj_1": {"name": "Test Project"}},
            "agent_performance": {"agent1": 0.8}
        }
        
        await service.save_state(str(test_file), test_data)
        assert test_file.exists()
        
        loaded_data = await service.load_state(str(test_file))
        assert loaded_data["agent_performance"]["agent1"] == 0.8
    
    @pytest.mark.asyncio
    async def test_checkpoint_management(self, service):
        """Test checkpoint creation and restoration."""
        test_data = {"test": "data"}
        
        await service.create_checkpoint("test_checkpoint", test_data)
        checkpoints = service.list_checkpoints()
        assert "test_checkpoint" in checkpoints
        
        restored_data = await service.restore_checkpoint("test_checkpoint")
        assert restored_data["test"] == "data"


class TestUnifiedTaskManagerFacade:
    """Test the facade integration."""
    
    @pytest.fixture
    def facade(self):
        """Create UnifiedTaskManagerFacade instance."""
        communication_protocol = Mock()
        decision_maker = Mock()
        return UnifiedTaskManagerFacade(
            communication_protocol=communication_protocol,
            decision_maker=decision_maker,
            num_agents=5,
            num_actions=10
        )
    
    @pytest.mark.asyncio
    async def test_facade_create_task(self, facade):
        """Test task creation through facade."""
        task = await facade.create_task("Test task", "agent1")
        
        assert task.description == "Test task"
        assert task.assigned_agents == ["agent1"]
        assert len(facade.pending_tasks) > 0
    
    @pytest.mark.asyncio
    async def test_facade_create_project(self, facade):
        """Test project creation through facade."""
        project_id = await facade.create_project("Test Project", "Description")
        
        assert project_id is not None
        projects = await facade.get_all_projects()
        assert project_id in projects
    
    @pytest.mark.asyncio
    async def test_facade_introspection(self, facade):
        """Test facade introspection functionality."""
        introspection = await facade.introspect()
        
        assert "pending_tasks" in introspection
        assert "ongoing_tasks" in introspection
        assert "completed_tasks" in introspection
        assert "projects" in introspection
        assert "available_agents" in introspection
        assert "batch_size" in introspection


if __name__ == "__main__":
    pytest.main([__file__, "-v"])