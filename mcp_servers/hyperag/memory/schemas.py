"""Storage schemas for HypeRAG dual-memory system"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class HippoSchema:
    """Schema definition for Hippo-Index (DuckDB)"""

    @staticmethod
    def get_create_tables_sql() -> list[str]:
        """Get SQL statements to create Hippo-Index tables"""
        return [
            # Episodic nodes table
            """
            CREATE TABLE IF NOT EXISTS hippo_nodes (
                id VARCHAR PRIMARY KEY,
                content TEXT NOT NULL,
                node_type VARCHAR NOT NULL,
                memory_type VARCHAR NOT NULL DEFAULT 'episodic',
                confidence REAL NOT NULL DEFAULT 1.0,
                embedding REAL[768],  -- Fixed-size embedding array
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                user_id VARCHAR,

                -- GDC support
                gdc_flags VARCHAR[],
                popularity_rank INTEGER DEFAULT 0,

                -- Temporal properties
                importance_score REAL DEFAULT 0.5,
                decay_rate REAL DEFAULT 0.1,
                ttl INTEGER,  -- seconds

                -- Uncertainty tracking
                uncertainty REAL DEFAULT 0.0,
                confidence_type VARCHAR DEFAULT 'temporal',

                -- Metadata as JSON
                metadata JSON
            )
            """,
            # Episodic edges table
            """
            CREATE TABLE IF NOT EXISTS hippo_edges (
                id VARCHAR PRIMARY KEY,
                source_id VARCHAR NOT NULL,
                target_id VARCHAR NOT NULL,
                relation VARCHAR NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,

                -- Hypergraph support
                participants VARCHAR[],  -- Array of participant node IDs

                -- Memory properties
                memory_type VARCHAR DEFAULT 'episodic',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP,
                access_count INTEGER DEFAULT 0,

                -- GDC and popularity
                gdc_flags VARCHAR[],
                popularity_rank INTEGER DEFAULT 0,

                -- Personalization (Rel-GAT)
                alpha_weight REAL,
                user_id VARCHAR,

                -- Uncertainty and evidence
                uncertainty REAL DEFAULT 0.0,
                evidence_count INTEGER DEFAULT 1,
                source_docs VARCHAR[],

                -- Tags and metadata
                tags VARCHAR[],
                metadata JSON,

                FOREIGN KEY (source_id) REFERENCES hippo_nodes(id),
                FOREIGN KEY (target_id) REFERENCES hippo_nodes(id)
            )
            """,
            # Document storage
            """
            CREATE TABLE IF NOT EXISTS hippo_documents (
                id VARCHAR PRIMARY KEY,
                content TEXT NOT NULL,
                doc_type VARCHAR NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                user_id VARCHAR,
                embedding REAL[768],
                confidence REAL DEFAULT 1.0,
                metadata JSON
            )
            """,
            # Consolidation tracking
            """
            CREATE TABLE IF NOT EXISTS consolidation_batches (
                id VARCHAR PRIMARY KEY,
                confidence_threshold REAL NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR DEFAULT 'pending',
                nodes_processed INTEGER DEFAULT 0,
                edges_processed INTEGER DEFAULT 0,
                metadata JSON
            )
            """,
        ]

    @staticmethod
    def get_create_indexes_sql() -> list[str]:
        """Get SQL statements to create indexes for performance"""
        return [
            # Node indexes
            "CREATE INDEX IF NOT EXISTS idx_hippo_nodes_embedding ON hippo_nodes USING HNSW(embedding)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_nodes_user_time ON hippo_nodes(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_nodes_confidence ON hippo_nodes(confidence DESC)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_nodes_importance ON hippo_nodes(importance_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_nodes_type ON hippo_nodes(node_type, memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_nodes_ttl ON hippo_nodes(created_at, ttl) WHERE ttl IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_hippo_nodes_gdc ON hippo_nodes USING GIN(gdc_flags)",
            # Edge indexes
            "CREATE INDEX IF NOT EXISTS idx_hippo_edges_source ON hippo_edges(source_id)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_edges_target ON hippo_edges(target_id)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_edges_relation ON hippo_edges(relation)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_edges_confidence ON hippo_edges(confidence DESC)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_edges_user ON hippo_edges(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_edges_participants ON hippo_edges USING GIN(participants)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_edges_popularity ON hippo_edges(popularity_rank DESC)",
            # Document indexes
            "CREATE INDEX IF NOT EXISTS idx_hippo_docs_type ON hippo_documents(doc_type)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_docs_user_time ON hippo_documents(user_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_hippo_docs_embedding ON hippo_documents USING HNSW(embedding)",
            # Consolidation indexes
            "CREATE INDEX IF NOT EXISTS idx_consolidation_status ON consolidation_batches(status, created_at)",
        ]

    @staticmethod
    def get_materialized_views_sql() -> list[str]:
        """Get SQL for materialized views for common queries"""
        return [
            # Recent nodes by user
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS recent_nodes_by_user AS
            SELECT
                user_id,
                COUNT(*) as node_count,
                AVG(confidence) as avg_confidence,
                MAX(created_at) as last_activity
            FROM hippo_nodes
            WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
            GROUP BY user_id
            """,
            # Popular relations
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS popular_relations AS
            SELECT
                relation,
                COUNT(*) as edge_count,
                AVG(confidence) as avg_confidence,
                AVG(popularity_rank) as avg_popularity
            FROM hippo_edges
            GROUP BY relation
            ORDER BY edge_count DESC
            """,
            # Expiring nodes
            """
            CREATE MATERIALIZED VIEW IF NOT EXISTS expiring_nodes AS
            SELECT
                id,
                content,
                user_id,
                created_at,
                ttl,
                (EXTRACT(EPOCH FROM CURRENT_TIMESTAMP - created_at)) as age_seconds
            FROM hippo_nodes
            WHERE ttl IS NOT NULL
            AND (EXTRACT(EPOCH FROM CURRENT_TIMESTAMP - created_at)) > (ttl * 0.8)
            ORDER BY age_seconds DESC
            """,
        ]


class HypergraphSchema:
    """Schema definition for Hypergraph-KG (Neo4j)"""

    @staticmethod
    def get_node_constraints() -> list[str]:
        """Get Cypher statements to create node constraints"""
        return [
            "CREATE CONSTRAINT semantic_node_id IF NOT EXISTS FOR (n:SemanticNode) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT hyperedge_id IF NOT EXISTS FOR (h:Hyperedge) REQUIRE h.id IS UNIQUE",
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        ]

    @staticmethod
    def get_relationship_constraints() -> list[str]:
        """Get Cypher statements for relationship constraints"""
        return [
            # Ensure hyperedge relationships have valid confidence
            """
            CREATE CONSTRAINT valid_confidence IF NOT EXISTS
            FOR ()-[r:PARTICIPATES]-()
            REQUIRE 0.0 <= r.confidence <= 1.0
            """,
            # Ensure consolidation tracking
            """
            CREATE CONSTRAINT consolidation_timestamp IF NOT EXISTS
            FOR ()-[r:CONSOLIDATED_FROM]-()
            REQUIRE r.consolidated_at IS NOT NULL
            """,
        ]

    @staticmethod
    def get_indexes() -> list[str]:
        """Get Cypher statements to create indexes"""
        return [
            # Node property indexes
            "CREATE INDEX semantic_node_confidence IF NOT EXISTS FOR (n:SemanticNode) ON (n.confidence)",
            "CREATE INDEX semantic_node_created IF NOT EXISTS FOR (n:SemanticNode) ON (n.created_at)",
            "CREATE INDEX semantic_node_importance IF NOT EXISTS FOR (n:SemanticNode) ON (n.importance_score)",
            "CREATE INDEX semantic_node_type IF NOT EXISTS FOR (n:SemanticNode) ON (n.node_type)",
            "CREATE INDEX semantic_node_user IF NOT EXISTS FOR (n:SemanticNode) ON (n.user_id)",
            # Entity indexes
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX entity_popularity IF NOT EXISTS FOR (e:Entity) ON (e.popularity_rank)",
            # Hyperedge indexes
            "CREATE INDEX hyperedge_relation IF NOT EXISTS FOR (h:Hyperedge) ON (h.relation)",
            "CREATE INDEX hyperedge_confidence IF NOT EXISTS FOR (h:Hyperedge) ON (h.confidence)",
            "CREATE INDEX hyperedge_created IF NOT EXISTS FOR (h:Hyperedge) ON (h.created_at)",
            "CREATE INDEX hyperedge_popularity IF NOT EXISTS FOR (h:Hyperedge) ON (h.popularity_rank)",
            # User indexes
            "CREATE INDEX user_created IF NOT EXISTS FOR (u:User) ON (u.created_at)",
            # Document indexes
            "CREATE INDEX document_type IF NOT EXISTS FOR (d:Document) ON (d.doc_type)",
            "CREATE INDEX document_created IF NOT EXISTS FOR (d:Document) ON (d.created_at)",
            # Composite indexes for common queries
            "CREATE INDEX node_user_confidence IF NOT EXISTS FOR (n:SemanticNode) ON (n.user_id, n.confidence)",
            "CREATE INDEX edge_relation_confidence IF NOT EXISTS FOR (h:Hyperedge) ON (h.relation, h.confidence)",
            # Full-text search indexes
            "CREATE FULLTEXT INDEX semantic_content IF NOT EXISTS FOR (n:SemanticNode) ON EACH [n.content]",
            "CREATE FULLTEXT INDEX entity_content IF NOT EXISTS FOR (e:Entity) ON EACH [e.content]",
            "CREATE FULLTEXT INDEX document_content IF NOT EXISTS FOR (d:Document) ON EACH [d.content]",
        ]

    @staticmethod
    def get_sample_data_cypher() -> list[str]:
        """Get Cypher statements to create sample data structure"""
        return [
            # Create semantic nodes
            """
            MERGE (ai:SemanticNode {
                id: 'semantic_ai_001',
                content: 'Artificial Intelligence',
                node_type: 'concept',
                confidence: 0.95,
                created_at: datetime(),
                importance_score: 0.9,
                popularity_rank: 1,
                gdc_flags: [],
                uncertainty: 0.05
            })
            """,
            """
            MERGE (ml:SemanticNode {
                id: 'semantic_ml_001',
                content: 'Machine Learning',
                node_type: 'concept',
                confidence: 0.92,
                created_at: datetime(),
                importance_score: 0.85,
                popularity_rank: 2,
                gdc_flags: [],
                uncertainty: 0.08
            })
            """,
            # Create hyperedge
            """
            MERGE (h:Hyperedge {
                id: 'hyperedge_001',
                relation: 'is_subfield_of',
                confidence: 0.9,
                created_at: datetime(),
                participants: ['semantic_ml_001', 'semantic_ai_001'],
                popularity_rank: 1,
                evidence_count: 5,
                gdc_flags: [],
                uncertainty: 0.1
            })
            """,
            # Create relationships
            """
            MATCH (ml:SemanticNode {id: 'semantic_ml_001'})
            MATCH (ai:SemanticNode {id: 'semantic_ai_001'})
            MATCH (h:Hyperedge {id: 'hyperedge_001'})
            MERGE (ml)-[:PARTICIPATES {role: 'subject', confidence: 0.9}]->(h)
            MERGE (ai)-[:PARTICIPATES {role: 'object', confidence: 0.9}]->(h)
            """,
            # Create user
            """
            MERGE (u:User {
                id: 'user_001',
                name: 'Example User',
                created_at: datetime(),
                alpha_profile: {
                    'is_subfield_of': 1.2,
                    'relates_to': 0.8,
                    'influences': 1.1
                }
            })
            """,
        ]


class QdrantSchema:
    """Schema for Qdrant vector collections"""

    @staticmethod
    def get_collection_configs() -> dict[str, dict[str, Any]]:
        """Get Qdrant collection configurations"""
        return {
            "hippo_embeddings": {
                "vectors": {"size": 768, "distance": "Cosine"},
                "payload_schema": {
                    "node_id": "keyword",
                    "content": "text",
                    "user_id": "keyword",
                    "confidence": "float",
                    "created_at": "datetime",
                    "memory_type": "keyword",
                    "gdc_flags": "keyword",
                    "importance_score": "float",
                },
            },
            "semantic_embeddings": {
                "vectors": {"size": 768, "distance": "Cosine"},
                "payload_schema": {
                    "node_id": "keyword",
                    "content": "text",
                    "confidence": "float",
                    "created_at": "datetime",
                    "node_type": "keyword",
                    "popularity_rank": "integer",
                    "community_id": "keyword",
                },
            },
            "user_profiles": {
                "vectors": {
                    "size": 256,  # Smaller for user preference vectors
                    "distance": "Cosine",
                },
                "payload_schema": {
                    "user_id": "keyword",
                    "interaction_count": "integer",
                    "last_updated": "datetime",
                    "alpha_weights": "text",  # JSON string
                },
            },
        }

    @staticmethod
    def get_hnsw_configs() -> dict[str, dict[str, Any]]:
        """Get HNSW index configurations for collections"""
        return {
            "hippo_embeddings": {
                "m": 16,
                "ef_construct": 200,
                "full_scan_threshold": 10000,
            },
            "semantic_embeddings": {
                "m": 32,  # Higher for better recall on semantic data
                "ef_construct": 400,
                "full_scan_threshold": 20000,
            },
            "user_profiles": {
                "m": 8,  # Lower for smaller collection
                "ef_construct": 100,
                "full_scan_threshold": 5000,
            },
        }


class RedisSchema:
    """Schema for Redis caching layer"""

    @staticmethod
    def get_key_patterns() -> dict[str, str]:
        """Get Redis key patterns for different data types"""
        return {
            # Node caching
            "node": "hyperag:node:{node_id}",
            "nodes_by_user": "hyperag:nodes:user:{user_id}",
            "recent_nodes": "hyperag:nodes:recent:{time_window}",
            # Edge caching
            "edge": "hyperag:edge:{edge_id}",
            "edges_by_relation": "hyperag:edges:relation:{relation}",
            "popular_edges": "hyperag:edges:popular",
            # Query result caching
            "query_result": "hyperag:query:{query_hash}",
            "similarity_cache": "hyperag:similarity:{embedding_hash}",
            # User profile caching
            "user_profile": "hyperag:user:{user_id}:profile",
            "alpha_weights": "hyperag:user:{user_id}:alpha",
            # System state
            "consolidation_lock": "hyperag:consolidation:lock",
            "last_consolidation": "hyperag:consolidation:last",
            "system_metrics": "hyperag:metrics",
            # GDC caching
            "gdc_violations": "hyperag:gdc:violations",
            "popularity_ranks": "hyperag:popularity:ranks",
        }

    @staticmethod
    def get_ttl_configs() -> dict[str, int]:
        """Get TTL configurations for different data types (seconds)"""
        return {
            "query_result": 3600,  # 1 hour
            "similarity_cache": 7200,  # 2 hours
            "user_profile": 86400,  # 24 hours
            "popular_edges": 21600,  # 6 hours
            "recent_nodes": 1800,  # 30 minutes
            "system_metrics": 300,  # 5 minutes
            "gdc_violations": 43200,  # 12 hours
            "consolidation_lock": 7200,  # 2 hours max lock
        }
