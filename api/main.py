from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from typing import Any, Dict, List, Optional
import jwt
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel
import asyncio
from rag_system.error_handling.error_handler import error_handler
from rag_system.core.performance_optimization import performance_optimizer

app = FastAPI(
    title="AI Village API",
    description="API for interacting with the AI Village system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = "your-secret-key"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    disabled: Optional[bool] = None

class TaskRequest(BaseModel):
    description: str
    priority: Optional[int] = 1
    deadline: Optional[datetime] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None

# Authentication
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return User(username=username)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # In production, validate against user database
    if form_data.username != "test" or form_data.password != "test":
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(
        data={"sub": form_data.username}
    )
    return {"access_token": access_token, "token_type": "bearer"}

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Task Management Endpoints
@app.post("/tasks", response_model=TaskResponse)
@performance_optimizer.monitor_resource_usage("api")
async def create_task(
    task: TaskRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new task."""
    try:
        # Implementation would delegate to task management system
        task_id = "task_123"  # Generated task ID
        return {
            "task_id": task_id,
            "status": "created",
            "result": None
        }
    except Exception as e:
        error_handler.log_error(e, {"component": "api", "operation": "create_task"})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}", response_model=TaskResponse)
@performance_optimizer.monitor_resource_usage("api")
async def get_task(
    task_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get task status and result."""
    try:
        # Implementation would fetch from task management system
        return {
            "task_id": task_id,
            "status": "completed",
            "result": {"data": "example result"}
        }
    except Exception as e:
        error_handler.log_error(e, {"component": "api", "operation": "get_task"})
        raise HTTPException(status_code=500, detail=str(e))

# System Status Endpoints
@app.get("/status")
@performance_optimizer.monitor_resource_usage("api")
async def get_system_status(current_user: User = Depends(get_current_user)):
    """Get system status and metrics."""
    try:
        return {
            "status": "operational",
            "metrics": performance_optimizer.get_optimization_stats(),
            "errors": error_handler.get_error_metrics()
        }
    except Exception as e:
        error_handler.log_error(e, {"component": "api", "operation": "get_status"})
        raise HTTPException(status_code=500, detail=str(e))

# Agent Interaction Endpoints
@app.post("/agents/{agent_id}/query")
@performance_optimizer.monitor_resource_usage("api")
async def query_agent(
    agent_id: str,
    query: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Send query to specific agent."""
    try:
        # Implementation would delegate to agent system
        return {
            "agent_id": agent_id,
            "response": "Agent response example"
        }
    except Exception as e:
        error_handler.log_error(e, {"component": "api", "operation": "query_agent"})
        raise HTTPException(status_code=500, detail=str(e))

# Knowledge Management Endpoints
@app.get("/knowledge/{concept}")
@performance_optimizer.cache_result(ttl=3600)
@performance_optimizer.monitor_resource_usage("api")
async def get_knowledge(
    concept: str,
    current_user: User = Depends(get_current_user)
):
    """Get knowledge about specific concept."""
    try:
        # Implementation would query knowledge management system
        return {
            "concept": concept,
            "knowledge": "Example knowledge"
        }
    except Exception as e:
        error_handler.log_error(e, {"component": "api", "operation": "get_knowledge"})
        raise HTTPException(status_code=500, detail=str(e))

# Tool Management Endpoints
@app.post("/tools")
@performance_optimizer.monitor_resource_usage("api")
async def create_tool(
    tool_spec: Dict[str, Any],
    current_user: User = Depends(get_current_user)
):
    """Create new tool."""
    try:
        # Implementation would delegate to tool management system
        return {
            "tool_id": "tool_123",
            "status": "created"
        }
    except Exception as e:
        error_handler.log_error(e, {"component": "api", "operation": "create_tool"})
        raise HTTPException(status_code=500, detail=str(e))

# Error Handling
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    error_handler.log_error(exc, {"component": "api", "path": request.url.path})
    return {"detail": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
@app.get("/rag/graph")
@performance_optimizer.cache_result(ttl=300)
async def get_knowledge_graph(current_user: User = Depends(get_current_user)):
    """Get knowledge graph data for visualization."""
    try:
        # Implementation would fetch from RAG system
        return {
            "nodes": [
                {"id": "concept1", "label": "Concept 1", "type": "concept"},
                {"id": "concept2", "label": "Concept 2", "type": "concept"},
                # Add more nodes
            ],
            "edges": [
                {"from": "concept1", "to": "concept2", "label": "relates_to"},
                # Add more edges
            ]
        }
    except Exception as e:
        error_handler.log_error(e, {"component": "api", "operation": "get_knowledge_graph"})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/king/decision-tree")
@performance_optimizer.cache_result(ttl=300)
async def get_decision_tree(current_user: User = Depends(get_current_user)):
    """Get King's decision tree for visualization."""
    try:
        # Implementation would fetch from King's planning system
        return {
            "nodes": [
                {"id": "goal1", "label": "Main Goal", "type": "goal"},
                {"id": "subgoal1", "label": "Sub-Goal 1", "type": "subgoal"},
                # Add more nodes
            ],
            "edges": [
                {"from": "goal1", "to": "subgoal1", "label": "requires"},
                # Add more edges
            ]
        }
    except Exception as e:
        error_handler.log_error(e, {"component": "api", "operation": "get_decision_tree"})
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/message")
@performance_optimizer.monitor_resource_usage("api")
async def send_chat_message(
    message: Dict[str, str],
    current_user: User = Depends(get_current_user)
):
    """Send message to chat system."""
    try:
        # Implementation would process message through agents
        return {
            "response": "Agent response",
            "context": {
                "relevant_concepts": ["concept1", "concept2"],
                "confidence": 0.95,
                "reasoning_path": ["step1", "step2"]
            }
        }
    except Exception as e:
        error_handler.log_error(e, {"component": "api", "operation": "send_chat_message"})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history")
@performance_optimizer.monitor_resource_usage("api")
async def get_chat_history(
    current_user: User = Depends(get_current_user),
    limit: int = 50
):
    """Get chat history."""
    try:
        # Implementation would fetch from chat history
        return {
            "messages": [
                {
                    "id": "msg1",
                    "sender": "user",
                    "content": "User message",
                    "timestamp": "2023-01-01T00:00:00Z"
                },
                {
                    "id": "msg2",
                    "sender": "agent",
                    "content": "Agent response",
                    "timestamp": "2023-01-01T00:00:01Z"
                }
            ]
        }
    except Exception as e:
        error_handler.log_error(e, {"component": "api", "operation": "get_chat_history"})
        raise HTTPException(status_code=500, detail=str(e))
