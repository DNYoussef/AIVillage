import asyncio
from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from ai_village.pipeline.ai_village_pipeline import AIVillagePipeline
from ai_village.utils.config import Config
from ai_village.utils.logger import setup_logger

app = FastAPI()
config = Config()
logger = setup_logger("ai_village")
pipeline = AIVillagePipeline(config)

class Query(BaseModel):
    text: str

@app.post("/query")
async def query(query: Query):
    try:
        result = await pipeline.process_query({"query": query.text})
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = await pipeline.upload_file(contents, file.filename)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/import_open_researcher")
async def import_open_researcher(data_path: str):
    try:
        result = await pipeline.import_open_researcher_data(data_path)
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error importing Open Researcher data: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)