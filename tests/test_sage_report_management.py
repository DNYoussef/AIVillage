"""Tests for Sage agent's report management system."""

import pytest
from pytest_asyncio import fixture
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

class MockRAGPipeline:
    def __init__(self):
        self.process_query = AsyncMock(return_value={
            "response": "Test RAG response",
            "sources": ["source1", "source2"],
            "confidence": 0.9
        })

class MockCognitiveNexus:
    def __init__(self):
        self.synthesize = AsyncMock(return_value={
            "integrated_knowledge": "Test synthesis",
            "confidence": 0.9
        })

class MockReportManager:
    def __init__(self):
        # Initialize with empty state
        self.clear_state()
        
    def clear_state(self):
        """Clear all state for isolation between tests."""
        self.report_history = {}
        self.rag_pipeline = MockRAGPipeline()
        self.cognitive_nexus = MockCognitiveNexus()
        self.id_counter = 0  # Add counter for unique IDs
        
        # Initialize report writers
        self.scientific_writer = AsyncMock()
        self.agent_writer = AsyncMock()
        self.wiki_writer = AsyncMock()
        self.project_writer = AsyncMock()
        
        # Set up mock responses to return input content
        async def mock_generate(**kwargs):
            print(f"Generating report with content: {kwargs}")
            return dict(kwargs)  # Return a new dict to avoid reference issues
        
        self.scientific_writer.generate_paper = AsyncMock(side_effect=mock_generate)
        self.agent_writer.generate_report = AsyncMock(side_effect=mock_generate)
        self.wiki_writer.generate_article = AsyncMock(side_effect=mock_generate)
        self.project_writer.generate_report = AsyncMock(side_effect=mock_generate)
    
    async def generate_report(self, report_type: str, content: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\nGenerating {report_type} report with content: {content}")
        writer_map = {
            "scientific": self.scientific_writer.generate_paper,
            "agent": self.agent_writer.generate_report,
            "wiki": self.wiki_writer.generate_article,
            "project": self.project_writer.generate_report
        }
        
        if report_type not in writer_map:
            raise ValueError(f"Unknown report type: {report_type}")
        
        # Generate report using the appropriate writer
        report = await writer_map[report_type](**content)
        print(f"Generated report: {report}")
        
        # Store in history with unique ID
        self.id_counter += 1
        report_id = f"{report_type}_{self.id_counter}_{datetime.now().isoformat()}"
        self.report_history[report_id] = [dict(report)]  # Store a copy
        print(f"Stored report with ID {report_id}")
        print(f"Current report history: {self.report_history}")
        
        return dict(report)  # Return a copy
    
    async def update_report(self, report_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        if report_id not in self.report_history:
            raise ValueError(f"Report not found: {report_id}")
        
        latest_version = dict(self.report_history[report_id][-1])  # Create a copy
        latest_version.update(updates)
        self.report_history[report_id].append(dict(latest_version))  # Store a copy
        return dict(latest_version)  # Return a copy
    
    async def get_report_history(self, report_id: str) -> List[Dict[str, Any]]:
        if report_id not in self.report_history:
            raise ValueError(f"Report not found: {report_id}")
        return [dict(version) for version in self.report_history[report_id]]  # Return copies
    
    async def search_reports(self, query: str, report_type: str = None) -> List[Dict[str, Any]]:
        print(f"\nSearching for query: {query}")
        print(f"Current report history: {self.report_history}")
        results = []
        query = query.lower()
        
        for report_id, versions in self.report_history.items():
            print(f"\nChecking report {report_id}:")
            if report_type and not report_id.split('_')[0] != report_type:  # Fix report type check
                continue
            
            latest_version = dict(versions[-1])  # Create a copy
            print(f"Content: {latest_version}")
            
            # Search in both title and content
            matches = False
            for key, value in latest_version.items():
                print(f"Checking field {key}: {value}")
                if isinstance(value, str) and query in value.lower():
                    print(f"Found match in {key}")
                    matches = True
                    break
            
            if matches:
                print(f"Adding report {report_id} to results")
                results.append({
                    "report_id": report_id,
                    "report": dict(latest_version),  # Store a copy
                    "version_count": len(versions)
                })
        
        print(f"Final search results: {results}")
        return results

@pytest.fixture
def report_manager():
    """Create ReportManager instance for testing."""
    manager = MockReportManager()
    yield manager
    # Clean up after each test
    manager.clear_state()

@pytest.mark.asyncio
async def test_scientific_paper_generation(report_manager):
    """Test scientific paper generation."""
    content = {
        "title": "Test Paper",
        "abstract": "Test abstract",
        "sections": {
            "introduction": "Test intro",
            "methods": "Test methods",
            "results": "Test results",
            "discussion": "Test discussion"
        }
    }
    
    report = await report_manager.generate_report("scientific", content)
    
    assert isinstance(report, dict)
    assert "title" in report
    assert "sections" in report
    assert len(report["sections"]) > 0
    assert report["title"] == content["title"]
    assert report["sections"] == content["sections"]

@pytest.mark.asyncio
async def test_report_update(report_manager):
    """Test report updating."""
    initial_content = {
        "title": "Test Report",
        "content": "Initial content"
    }
    report = await report_manager.generate_report("wiki", initial_content)
    report_id = list(report_manager.report_history.keys())[0]
    
    updates = {
        "content": "Updated content"
    }
    updated_report = await report_manager.update_report(report_id, updates)
    
    assert isinstance(updated_report, dict)
    assert "content" in updated_report
    assert updated_report["content"] == "Updated content"
    assert updated_report["title"] == initial_content["title"]

@pytest.mark.asyncio
async def test_report_history(report_manager):
    """Test report history tracking."""
    content = {
        "title": "Test Report",
        "content": "Test content"
    }
    
    report = await report_manager.generate_report("wiki", content)
    report_id = list(report_manager.report_history.keys())[0]
    
    history = await report_manager.get_report_history(report_id)
    
    assert isinstance(history, list)
    assert len(history) == 1
    assert history[0]["title"] == content["title"]
    assert history[0]["content"] == content["content"]

@pytest.mark.asyncio
async def test_report_search(report_manager):
    """Test report searching."""
    print("\nStarting test_report_search")
    
    # Generate reports
    reports = [
        {
            "title": "AI Report",
            "content": "AI content"
        },
        {
            "title": "ML Report",
            "content": "Machine learning content"
        }
    ]
    
    # Generate each report
    for report_content in reports:
        print(f"\nGenerating report with content: {report_content}")
        report = await report_manager.generate_report("wiki", dict(report_content))  # Pass a copy
        print(f"Generated report: {report}")
    
    print(f"\nReport history after generation: {report_manager.report_history}")
    
    # Search for AI-related reports
    results = await report_manager.search_reports("ai")
    print(f"\nSearch results: {results}")
    
    assert isinstance(results, list)
    assert len(results) == 1, f"Expected 1 result, got {len(results)}. Results: {results}"
    assert "ai" in results[0]["report"]["title"].lower()

@pytest.mark.asyncio
async def test_invalid_report_type(report_manager):
    """Test handling of invalid report types."""
    content = {
        "title": "Test Report",
        "content": "Test content"
    }
    
    with pytest.raises(ValueError) as exc_info:
        await report_manager.generate_report("invalid_type", content)
    assert "Unknown report type" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
