# Interface Standardization Migration Guide

This guide demonstrates how to migrate existing AIVillage components to use the standardized interfaces, addressing Foundation Issue #2: "Ensure consistent interfaces across all components and agent types."

## Overview

The new interface framework provides:
- **Consistent Agent Interface**: Standard methods and capabilities for all agents
- **Communication Protocol**: Unified messaging and protocol handling
- **RAG Interface**: Standardized retrieval and generation operations
- **Processing Interface**: Common processing patterns and queue management
- **Training Interface**: Unified training pipeline and model management

## Interface Hierarchy

```
┌─ AgentInterface (Base for all agents)
├─ CommunicationInterface (Message handling)
├─ RAGInterface (Retrieval-augmented generation)
├─ ProcessingInterface (General processing)
└─ TrainingInterface (Model training)
```

## Migration Steps

### Step 1: Import Standard Interfaces

```python
from agents.interfaces import (
    AgentInterface,
    AgentMetadata,
    AgentCapability,
    TaskInterface,
    MessageInterface,
    CommunicationInterface,
    RAGInterface,
    ProcessingInterface,
    TrainingInterface
)
```

### Step 2: Update Agent Classes

#### Before (agents/king/king_agent.py):
```python
class KingAgent:
    def __init__(self, config):
        self.config = config
        self.name = "King"
        self.capabilities = ["coordination", "task_management"]

    async def process_task(self, task):
        # Custom implementation
        pass

    async def send_message(self, recipient, message):
        # Custom implementation
        pass
```

#### After:
```python
from agents.interfaces import (
    AgentInterface,
    AgentMetadata,
    AgentCapability,
    TaskInterface,
    MessageInterface
)

class KingAgent(AgentInterface):
    def __init__(self, config):
        # Create standardized metadata
        metadata = AgentMetadata(
            agent_id="king-001",
            agent_type="CoordinatorAgent",
            name="King",
            description="Main coordination agent for task management",
            version="1.0.0",
            capabilities={
                AgentCapability.TASK_EXECUTION,
                AgentCapability.INTER_AGENT_COMMUNICATION,
                AgentCapability.PLANNING,
                AgentCapability.DECISION_MAKING
            }
        )

        super().__init__(metadata)
        self.config = config

    async def initialize(self) -> bool:
        """Initialize King agent."""
        self.set_status(AgentStatus.IDLE)
        return True

    async def shutdown(self) -> bool:
        """Shutdown King agent."""
        self.set_status(AgentStatus.OFFLINE)
        return True

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "agent_id": self.metadata.agent_id,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.get_capabilities()],
            "tasks_processed": self.performance_metrics.total_tasks_processed,
            "success_rate": self.performance_metrics.success_rate
        }

    async def process_task(self, task: TaskInterface) -> Dict[str, Any]:
        """Process task with standardized interface."""

        # Update metrics
        start_time = time.time()

        try:
            # Process the task
            result = await self._process_task_internal(task)

            # Update success metrics
            processing_time = (time.time() - start_time) * 1000
            self.update_performance_metrics(
                total_tasks_processed=self.performance_metrics.total_tasks_processed + 1,
                successful_tasks=self.performance_metrics.successful_tasks + 1,
                average_processing_time_ms=processing_time
            )

            return {
                "task_id": task.task_id,
                "result": result,
                "status": "completed",
                "processing_time_ms": processing_time
            }

        except Exception as e:
            # Update failure metrics
            processing_time = (time.time() - start_time) * 1000
            self.update_performance_metrics(
                total_tasks_processed=self.performance_metrics.total_tasks_processed + 1,
                failed_tasks=self.performance_metrics.failed_tasks + 1
            )

            raise AgentException(
                f"Task processing failed: {str(e)}",
                agent_name=self.metadata.name,
                context=self.create_error_context("process_task", task_id=task.task_id)
            )

    async def can_handle_task(self, task: TaskInterface) -> bool:
        """Check if King can handle this task type."""
        return task.task_type in ["coordination", "planning", "decision"]

    async def estimate_task_duration(self, task: TaskInterface) -> Optional[float]:
        """Estimate task processing time."""
        # Simple estimation based on task type
        estimates = {
            "coordination": 2.0,
            "planning": 5.0,
            "decision": 1.0
        }
        return estimates.get(task.task_type, 3.0)

    async def send_message(self, message: MessageInterface) -> bool:
        """Send message using communication interface."""
        # Implementation depends on communication system
        return await self._communication_handler.send_message(message)

    async def receive_message(self, message: MessageInterface) -> None:
        """Receive and process message."""
        # Handle different message types
        if message.message_type == "task_assignment":
            task = TaskInterface.from_dict(message.content)
            await self.process_task(task)
        elif message.message_type == "status_request":
            response = await self.health_check()
            # Send response back

    async def broadcast_message(self, message: MessageInterface, recipients: List[str]) -> Dict[str, bool]:
        """Broadcast message to multiple agents."""
        return await self._communication_handler.broadcast_message(message, recipients)
```

### Step 3: Update RAG Systems

#### Before (rag_system/core/pipeline.py):
```python
class EnhancedRAGPipeline:
    async def process_query(self, query: str) -> Dict[str, Any]:
        # Custom implementation
        pass
```

#### After:
```python
from agents.interfaces import (
    RAGInterface,
    QueryInterface,
    DocumentInterface,
    QueryType,
    RetrievalStrategy
)

class EnhancedRAGPipeline(RAGInterface):
    async def initialize(self, document_store, embedding_model, **config) -> bool:
        """Initialize RAG pipeline."""
        self.document_store = document_store
        self.embedding_model = embedding_model
        self.config = config
        return True

    async def process_query(self, query: QueryInterface) -> QueryResult:
        """Process query with standardized interface."""

        start_time = time.time()

        try:
            # Retrieve relevant documents
            retrieved_docs = await self.retrieve_documents(query)

            # Generate response
            response = await self.generate_response(
                query.query_text,
                [result.document for result in retrieved_docs]
            )

            processing_time = (time.time() - start_time) * 1000

            # Calculate confidence based on retrieval scores
            avg_relevance = sum(r.relevance_score for r in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0.0

            return QueryResult(
                query=query,
                retrieved_documents=retrieved_docs,
                generated_response=response,
                confidence_score=avg_relevance,
                processing_time_ms=processing_time,
                sources=[doc.document.metadata.source for doc in retrieved_docs]
            )

        except Exception as e:
            raise RAGException(
                f"Query processing failed: {str(e)}",
                context=self.create_error_context("process_query", query_id=query.query_id)
            )

    async def retrieve_documents(self, query: QueryInterface) -> List[RetrievalResult]:
        """Retrieve relevant documents."""
        return await self.document_store.search_documents(query)

    async def generate_response(self, query: str, context_docs: List[DocumentInterface]) -> str:
        """Generate response using retrieved documents."""
        # Combine document content for context
        context = "\n".join([doc.content[:500] for doc in context_docs])

        # Generate response (implementation depends on LLM)
        response = await self._llm.generate(
            prompt=f"Question: {query}\nContext: {context}\nAnswer:"
        )

        return response

    async def add_documents(self, documents: List[DocumentInterface]) -> Dict[str, bool]:
        """Add documents to RAG system."""
        results = {}

        for doc in documents:
            try:
                # Generate embeddings if not present
                if not doc.embeddings and doc.chunks:
                    embeddings = await self.embedding_model.encode_batch(doc.chunks)
                    doc.embeddings = embeddings

                # Add to document store
                success = await self.document_store.add_document(doc)
                results[doc.document_id] = success

            except Exception as e:
                results[doc.document_id] = False

        return results
```

### Step 4: Update Communication Systems

#### Before (communications/protocol.py):
```python
class StandardCommunicationProtocol:
    async def send_message(self, message):
        # Custom implementation
        pass
```

#### After:
```python
from agents.interfaces import (
    CommunicationInterface,
    MessageProtocol,
    ProtocolConfig,
    ProtocolCapability,
    MessageInterface
)

class StandardCommunicationProtocol(CommunicationInterface):
    async def initialize(self, protocols: List[MessageProtocol]) -> bool:
        """Initialize communication system."""

        for protocol in protocols:
            await protocol.initialize()
            self.protocols[protocol.config.protocol_name] = protocol

        # Set default protocol
        if protocols:
            self.default_protocol = protocols[0].config.protocol_name

        return True

    async def shutdown(self) -> bool:
        """Shutdown communication system."""
        for protocol in self.protocols.values():
            await protocol.shutdown()

        self.protocols.clear()
        return True

    # Inherited methods from CommunicationInterface provide the rest
```

### Step 5: Update Training Systems

#### Before (agent_forge/training/training_loop.py):
```python
class TrainingLoop:
    def train_epoch(self, model, data):
        # Custom implementation
        pass
```

#### After:
```python
from agents.interfaces import (
    TrainingInterface,
    ModelInterface,
    TrainingConfig,
    TrainingStatus,
    TrainingPhase
)

class AgentForgeTrainingLoop(TrainingInterface):
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.geometric_monitor = None

        if config.enable_geometric_monitoring:
            from agent_forge.geometry.snapshot import snapshot
            self.geometric_monitor = snapshot

    async def initialize(self, model: ModelInterface) -> bool:
        """Initialize training with model."""
        self.model = model
        self.set_status(TrainingStatus.READY)

        # Initialize geometric monitoring if enabled
        if self.config.enable_geometric_monitoring:
            self.add_callback(self._geometric_callback)

        return True

    async def train_epoch(self, epoch: int) -> TrainingMetrics:
        """Train single epoch with geometric monitoring."""

        self.set_status(TrainingStatus.TRAINING)
        epoch_start = time.time()

        total_loss = 0.0
        total_samples = 0

        # Training loop
        for batch_idx, batch in enumerate(self.data_loader):
            # Forward pass
            outputs = await self.model.forward(batch.inputs)
            loss = self.criterion(outputs, batch.targets)

            # Geometric monitoring
            if self.config.enable_geometric_monitoring and hasattr(outputs, 'hidden_states'):
                geom_metrics = await self.monitor_geometry(outputs.hidden_states)

                # Check for grokking
                if geom_metrics.get("intrinsic_dimensionality", 1.0) < self.config.grokking_threshold:
                    self.metrics.grokking_detected = True

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            total_samples += len(batch.targets)

        # Update metrics
        epoch_time = time.time() - epoch_start

        self.update_metrics(
            epoch=epoch,
            loss=total_loss / len(self.data_loader),
            training_time_seconds=epoch_time,
            samples_processed=total_samples
        )

        return self.metrics

    async def validate(self) -> TrainingMetrics:
        """Run validation."""
        self.set_status(TrainingStatus.EVALUATING)

        val_loss = 0.0
        val_accuracy = 0.0

        # Validation loop
        for batch in self.val_loader:
            outputs = await self.model.forward(batch.inputs)
            loss = self.criterion(outputs, batch.targets)

            val_loss += loss.item()
            # Calculate accuracy...

        # Update validation metrics
        self.update_metrics(
            validation_loss=val_loss / len(self.val_loader),
            validation_accuracy=val_accuracy
        )

        self.set_status(TrainingStatus.TRAINING)
        return self.metrics

    async def _geometric_callback(self, event: str, *args, **kwargs):
        """Callback for geometric monitoring."""
        if event == "epoch_end" and self.metrics.grokking_detected:
            print(f"Grokking detected at epoch {self.metrics.epoch}!")
```

## Advanced Interface Features

### Custom Agent Capabilities

```python
# Define custom capabilities
class CustomAgentCapability(Enum):
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CODE_REVIEW = "code_review"
    DOCUMENT_GENERATION = "document_generation"

# Extend agent with custom capabilities
class SpecializedAgent(AgentInterface):
    def __init__(self):
        metadata = AgentMetadata(
            agent_id="specialized-001",
            agent_type="SpecializedAgent",
            name="Specialized",
            description="Agent with custom capabilities",
            version="1.0.0",
            capabilities={
                AgentCapability.TEXT_PROCESSING,
                # Add custom capabilities as strings
            }
        )
        super().__init__(metadata)

        # Add custom capabilities
        self.metadata.capabilities.add("sentiment_analysis")
        self.metadata.capabilities.add("code_review")
```

### Message Routing

```python
class SmartCommunicationInterface(CommunicationInterface):
    async def route_message(self, message: MessageInterface) -> bool:
        """Smart message routing based on content."""

        # Route based on message type
        if message.message_type == "urgent":
            # Use high-priority protocol
            return await self.send_message(message, protocol="priority_protocol")
        elif message.message_type == "broadcast":
            # Use broadcast protocol
            recipients = await self.get_all_agents()
            return await self.broadcast_message(message, recipients)
        else:
            # Use default protocol
            return await self.send_message(message)
```

### RAG with Multiple Strategies

```python
class MultiStrategyRAG(RAGInterface):
    async def retrieve_documents(self, query: QueryInterface) -> List[RetrievalResult]:
        """Use multiple retrieval strategies."""

        results = []

        if query.retrieval_strategy == RetrievalStrategy.HYBRID:
            # Combine vector and keyword search
            vector_results = await self._vector_search(query)
            keyword_results = await self._keyword_search(query)

            # Merge and deduplicate results
            results = self._merge_results(vector_results, keyword_results)
        else:
            # Use single strategy
            results = await super().retrieve_documents(query)

        return results
```

## Migration Checklist

### Phase 1: Core Interfaces (Week 1)
- [ ] `agents/unified_base_agent.py` - Implement AgentInterface
- [ ] `agents/king/king_agent.py` - Migrate to AgentInterface
- [ ] `agents/sage/sage_agent.py` - Migrate to AgentInterface
- [ ] `communications/protocol.py` - Implement CommunicationInterface

### Phase 2: Specialized Interfaces (Week 2)
- [ ] `rag_system/core/pipeline.py` - Implement RAGInterface
- [ ] `agent_forge/training/training_loop.py` - Implement TrainingInterface
- [ ] Processing components - Implement ProcessingInterface
- [ ] All remaining agents - Implement AgentInterface

### Phase 3: Integration & Testing (Week 3)
- [ ] Integration testing with new interfaces
- [ ] Performance validation
- [ ] Documentation updates
- [ ] Legacy compatibility layer removal

## Benefits After Migration

1. **Consistent API**: All components use the same interface patterns
2. **Interoperability**: Components can be easily swapped and tested
3. **Type Safety**: Strong typing with interface validation
4. **Performance Monitoring**: Built-in metrics and health checks
5. **Error Handling**: Standardized error contexts and handling
6. **Testing**: Easier mocking and unit testing with standard interfaces
7. **Documentation**: Self-documenting interfaces with clear contracts

## Testing Interfaces

```python
import pytest
from agents.interfaces import validate_agent_interface

def test_agent_interface_compliance():
    agent = KingAgent(config)

    # Validate interface compliance
    assert validate_agent_interface(agent)

    # Test required methods
    assert hasattr(agent, 'process_task')
    assert hasattr(agent, 'send_message')
    assert hasattr(agent, 'health_check')

    # Test capabilities
    assert agent.has_capability(AgentCapability.TASK_EXECUTION)

@pytest.mark.asyncio
async def test_task_processing():
    agent = KingAgent(config)
    await agent.initialize()

    task = TaskInterface(
        task_id="test-001",
        task_type="coordination",
        content="Test coordination task"
    )

    result = await agent.process_task(task)
    assert result["status"] == "completed"
```

This interface standardization provides a solid foundation for consistent component behavior and easier system integration across the entire AIVillage platform.
