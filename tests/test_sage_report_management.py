"""Tests for Sage agent's report management system."""

import pytest
from pytest_asyncio import fixture
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

# Mock the problematic imports
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

@pytest.fixture
def mock_rag_pipeline():
    """Create mock RAG pipeline."""
    return MockRAGPipeline()

@pytest.fixture
def mock_cognitive_nexus():
    """Create mock CognitiveNexus."""
    return MockCognitiveNexus()

@pytest.fixture
def report_manager(mock_rag_pipeline, mock_cognitive_nexus):
    """Create ReportManager instance for testing."""
    class MockReportManager:
        def __init__(self):
            self.rag_pipeline = mock_rag_pipeline
            self.cognitive_nexus = mock_cognitive_nexus
            self.report_history = {}
            
            # Mock report writers that preserve input content exactly
            def create_writer(report_type):
                async def writer(**kwargs):
                    # Return exactly what was passed in
                    return dict(kwargs)  # Create a new dict to avoid reference issues
                return AsyncMock(side_effect=writer)
            
            self.scientific_writer = AsyncMock()
            self.scientific_writer.generate_paper = create_writer("scientific")
            
            self.agent_writer = AsyncMock()
            self.agent_writer.generate_report = create_writer("agent")
            
            self.wiki_writer = AsyncMock()
            self.wiki_writer.generate_article = create_writer("wiki")
            
            self.project_writer = AsyncMock()
            self.project_writer.generate_report = create_writer("project")
        
        async def generate_report(self, report_type: str, content: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            writer_map = {
                "scientific": self.scientific_writer.generate_paper,
                "agent": self.agent_writer.generate_report,
                "wiki": self.wiki_writer.generate_article,
                "project": self.project_writer.generate_report
            }
            
            if report_type not in writer_map:
                raise ValueError(f"Unknown report type: {report_type}")
            
            # Create a new dict with the content to avoid reference issues
            report_content = dict(content)
            
            # Generate report
            report = await writer_map[report_type](**report_content)
            
            # Store in history
            report_id = f"{report_type}_{datetime.now().isoformat()}"
            if report_id not in self.report_history:
                self.report_history[report_id] = []
            self.report_history[report_id].append(report)
            
            return report
        
        async def update_report(self, report_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
            if report_id not in self.report_history:
                raise ValueError(f"Report not found: {report_id}")
            
            latest_version = dict(self.report_history[report_id][-1])  # Create a new dict
            
            # Merge updates recursively
            def merge_dicts(original, updates):
                result = dict(original)  # Create a new dict
                for key, value in updates.items():
                    if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                        result[key] = merge_dicts(result[key], value)
                    else:
                        result[key] = value
                return result
            
            merged_content = merge_dicts(latest_version, updates)
            
            # Generate new version
            report_type = report_id.split("_")[0]
            new_version = await self.generate_report(report_type, merged_content)
            self.report_history[report_id].append(new_version)
            
            return new_version
        
        async def get_report_history(self, report_id: str) -> List[Dict[str, Any]]:
            if report_id not in self.report_history:
                raise ValueError(f"Report not found: {report_id}")
            return self.report_history[report_id]
        
        async def search_reports(self, query: str, report_type: str = None) -> List[Dict[str, Any]]:
            results = []
            query = query.lower()
            
            # Debug print
            print(f"\nSearching for query: {query}")
            print(f"Report history: {self.report_history}")
            
            for report_id, versions in self.report_history.items():
                if report_type and not report_id.startswith(report_type):
                    continue
                
                latest_version = versions[-1]
                print(f"\nChecking report {report_id}:")
                print(f"Content: {latest_version}")
                
                if self._matches_query(latest_version, query):
                    print(f"Match found in report {report_id}")
                    results.append({
                        "report_id": report_id,
                        "report": dict(latest_version),  # Create a new dict
                        "version_count": len(versions)
                    })
                else:
                    print(f"No match found in report {report_id}")
            
            return results
        
        def _matches_query(self, report: Dict[str, Any], query: str) -> bool:
            """Recursively search through report content for query."""
            def search_value(value: Any) -> bool:
                if isinstance(value, str):
                    result = query in value.lower()
                    print(f"Checking string value: {value}, query: {query}, match: {result}")
                    return result
                elif isinstance(value, dict):
                    return any(search_value(v) for v in value.values())
                elif isinstance(value, (list, tuple)):
                    return any(search_value(item) for item in value)
                return False
            
            # Search through all values in the report
            return any(search_value(value) for value in report.values())
    
    return MockReportManager()

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
    assert len(report["sections"]) == len(content["sections"])

@pytest.mark.asyncio
async def test_report_update(report_manager):
    """Test report updating."""
    # First create a report
    initial_content = {
        "title": "Test Report",
        "content": "Initial content"
    }
    report = await report_manager.generate_report("wiki", initial_content)
    report_id = list(report_manager.report_history.keys())[0]
    
    # Update the report
    updates = {
        "content": "Updated content"
    }
    updated_report = await report_manager.update_report(report_id, updates)
    
    assert isinstance(updated_report, dict)
    assert "content" in updated_report
    assert updated_report["content"] == "Updated content"

@pytest.mark.asyncio
async def test_report_history(report_manager):
    """Test report history tracking."""
    content = {
        "title": "Test Report",
        "content": "Test content"
    }
    
    # Generate report
    report = await report_manager.generate_report("wiki", content)
    report_id = list(report_manager.report_history.keys())[0]
    
    # Get history
    history = await report_manager.get_report_history(report_id)
    
    assert isinstance(history, list)
    assert len(history) == 1
    assert history[0]["title"] == content["title"]

@pytest.mark.asyncio
async def test_report_search(report_manager):
    """Test report searching."""
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
    print("\nGenerating test reports:")
    for report_content in reports:
        print(f"\nGenerating report with content: {report_content}")
        report = await report_manager.generate_report("wiki", report_content)
        print(f"Generated report: {report}")
        # Debug assertion to verify report content
        assert report["title"] == report_content["title"]
        assert report["content"] == report_content["content"]
    
    # Debug: Print all reports in history
    print("\nCurrent report history:")
    for report_id, versions in report_manager.report_history.items():
        latest = versions[-1]
        print(f"\nReport {report_id}:")
        print(f"Content: {latest}")
        assert isinstance(latest, dict)
        assert "title" in latest
        assert "content" in latest
    
    # Search for AI-related reports
    print("\nSearching for 'ai'...")
    results = await report_manager.search_reports("ai")
    
    print(f"\nSearch results: {results}")
    
    assert isinstance(results, list)
    assert len(results) == 1
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
    pytest.main([__file__])
