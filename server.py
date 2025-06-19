from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.error_handling.error_control import ErrorController
try:
    from rag_system.error_handling.hybrid_controller import HybridErrorController
except Exception:  # pragma: no cover
    HybridErrorController = None

app = FastAPI()

controller = HybridErrorController(4, 0.05, 0.1, 0.95) if HybridErrorController else ErrorController()
rag_pipeline = EnhancedRAGPipeline(error_controller=controller)
vector_store = rag_pipeline.hybrid_retriever.vector_store

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
