# rag_system/main.py

from core.config import RAGConfig
from core.pipeline import RAGPipeline
from data_management.document_manager import DocumentManager
from analysis.system_analyzer import SystemAnalyzer
from utils.logging import setup_logger

logger = setup_logger(__name__)

async def main():
    config = RAGConfig()
    pipeline = RAGPipeline(config)
    document_manager = DocumentManager(config)
    system_analyzer = SystemAnalyzer(config)
    
    # Example usage
    query = "Your query here"
    try:
        result = await pipeline.process_query(query)
        logger.info(f"Query result: {result}")

        # Document management example
        await document_manager.add_document(b"Document content", "example.txt")
        
        # System analysis example
        analysis_result = await system_analyzer.analyze_system_structure()
        logger.info(f"System analysis result: {analysis_result}")

    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
