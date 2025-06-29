from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, ValidationError
import os
from utils.logging import get_logger

from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.tracking.unified_knowledge_tracker import UnifiedKnowledgeTracker

logger = get_logger(__name__)

API_KEY = os.getenv("API_KEY")

class AuthMiddleware(BaseHTTPMiddleware):
    """Simple API key authentication and validation middleware."""
    async def dispatch(self, request: Request, call_next):
        if API_KEY and request.url.path not in ("/", "/ui", "/ui/index.html"):
            key = request.headers.get("x-api-key")
            if key != API_KEY:
                return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        try:
            response = await call_next(request)
        except ValidationError as exc:
            return JSONResponse(status_code=400, content={"detail": exc.errors()})
        return response

app = FastAPI(middleware=[Middleware(AuthMiddleware)])
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
