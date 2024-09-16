# rag_system/input_pipeline/input_pipeline.py

# ... (previous imports)

class InputPipeline:
    def __init__(self, config: Config, agent: AgentInterface, embedding_model: EmbeddingModel, 
                 vector_store: VectorStore, graph_store: GraphStore):
        self.config = config
        self.document_loader = DocumentLoader()
        self.document_processor = DocumentProcessor()
        self.agentic_chunker = PlanAwareAgenticChunker(ChatOpenAI(temperature=0, model_name=config.LLM_MODEL))
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.metadata_extractor = MetadataExtractor(agent)
        self.vector_store_inserter = VectorStoreInserter(vector_store)
        self.graph_constructor = GraphConstructor(graph_store, agent)
        self.concept_deduplicator = ConceptDeduplicator(config)
        self.code_parser = CodeParser()
        self.metadata_manager = MetadataManager(config)
        self.cache_manager = CacheManager(config)

    async def process(self, file_paths: Union[str, List[str]], current_plan: str):
        # 1. Load Documents
        documents = self.document_loader.load(file_paths)

        # 2. Preprocess Documents
        documents = self.document_processor.preprocess(documents)

        # 3. Parse Code (if applicable)
        documents = self.code_parser.parse(documents)

        # 4. Plan-Aware Agentic Chunking
        chunked_docs = self._plan_aware_chunk(documents, current_plan)

        # 5. Generate Embeddings
        chunked_docs = self.embedding_generator.generate_embeddings(chunked_docs)

        # 6. Extract Metadata
        chunked_docs = await self.metadata_extractor.extract_metadata(chunked_docs)

        # 7. Manage Metadata
        chunked_docs = self.metadata_manager.process_metadata(chunked_docs)

        # 8. Deduplicate Concepts
        chunked_docs = self.concept_deduplicator.deduplicate(chunked_docs)

        # 9. Insert into Vector Store
        await self.vector_store_inserter.insert(chunked_docs)

        # 10. Construct Graph
        await self.graph_constructor.construct_graph(chunked_docs)

        # 11. Cache processed documents
        self.cache_manager.cache_documents(chunked_docs)

        return chunked_docs

    def _plan_aware_chunk(self, documents: List[Document], current_plan: str) -> List[Document]:
        chunked_docs = []
        for doc in documents:
            propositions = self._split_into_propositions(doc.page_content)
            self.agentic_chunker.add_propositions(propositions, current_plan)
        
        for chunk in self.agentic_chunker.get_chunks('dict').values():
            chunked_doc = Document(
                page_content=" ".join(chunk['propositions']),
                metadata={
                    'chunk_id': chunk['chunk_id'],
                    'title': chunk['title'],
                    'summary': chunk['summary'],
                    'importance': chunk.get('importance', []),
                    'category': chunk.get('category', []),
                    'relationships': chunk.get('relationships', [])
                }
            )
            chunked_docs.append(chunked_doc)
        return chunked_docs

    def _split_into_propositions(self, text: str) -> List[str]:
        # This is a simple split by sentences. You might want to use a more sophisticated method.
        return [sent.strip() for sent in text.split('.') if sent.strip()]

    async def incremental_update(self, new_file_paths: Union[str, List[str]], current_plan: str):
        # Process new documents
        new_documents = await self.process(new_file_paths, current_plan)

        # Update existing knowledge graph
        await self.graph_constructor.update_graph(new_documents)

        # Rerank and update vector store
        await self.vector_store_inserter.rerank_and_update(new_documents)

        return new_documents
