from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker

app = FastAPI()
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

rag_pipeline = EnhancedRAGPipeline()
vector_store = rag_pipeline.hybrid_retriever.vector_store
knowledge_tracker = UnifiedKnowledgeTracker(
    rag_pipeline.hybrid_retriever.vector_store, rag_pipeline.hybrid_retriever.graph_store
)
rag_pipeline.knowledge_tracker = knowledge_tracker

class QueryRequest(BaseModel):
    query: str

@app.on_event("startup")
async def startup_event():
    await rag_pipeline.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await rag_pipeline.shutdown()

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    result = await rag_pipeline.process(request.query)
    return result

@app.post("/upload")
async def upload_endpoint(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    await vector_store.add_texts([text])
    return {"status": "uploaded"}

@app.get("/")
async def root():
    return FileResponse("ui/index.html")

@app.get("/status")
async def status_endpoint():
    return await rag_pipeline.get_status()

@app.get("/bayes")
async def bayes_endpoint():
    return rag_pipeline.get_bayes_net_snapshot()

@app.get("/logs")
async def logs_endpoint():
    return knowledge_tracker.retrieval_log
