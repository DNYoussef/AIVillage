"""Neo4j Database Migrations for Hypergraph Schema.

Handles schema creation, constraints, and indexes for the hypergraph knowledge system.
Includes both episodic (Hippo) and semantic (Hypergraph-KG) structures.
"""

import logging
from typing import Any

from neo4j import AsyncSession, Driver, Session

logger = logging.getLogger(__name__)


class HypergraphMigrations:
    """Manages Neo4j schema migrations for hypergraph system."""

    def __init__(self, driver: Driver) -> None:
        self.driver = driver
        self.migrations = self._define_migrations()

    def _define_migrations(self) -> list[dict[str, Any]]:
        """Define all schema migrations in order."""
        return [
            {
                "version": "001",
                "name": "create_hyperedge_constraints",
                "description": "Create constraints for hyperedge nodes",
                "up": [
                    "CREATE CONSTRAINT hyperedge_id IF NOT EXISTS FOR (h:Hyperedge) REQUIRE h.id IS UNIQUE",
                    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                    "CREATE CONSTRAINT hippo_node_id IF NOT EXISTS FOR (n:HippoNode) REQUIRE n.id IS UNIQUE",
                ],
                "down": [
                    "DROP CONSTRAINT hyperedge_id IF EXISTS",
                    "DROP CONSTRAINT entity_id IF EXISTS",
                    "DROP CONSTRAINT hippo_node_id IF EXISTS",
                ],
            },
            {
                "version": "002",
                "name": "create_performance_indexes",
                "description": "Create indexes for query performance",
                "up": [
                    "CREATE INDEX hyperedge_relation IF NOT EXISTS FOR (h:Hyperedge) ON (h.relation)",
                    "CREATE INDEX hyperedge_confidence IF NOT EXISTS FOR (h:Hyperedge) ON (h.confidence)",
                    "CREATE INDEX hyperedge_timestamp IF NOT EXISTS FOR (h:Hyperedge) ON (h.timestamp)",
                    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                    "CREATE INDEX hippo_session IF NOT EXISTS FOR (n:HippoNode) ON (n.session_id)",
                    "CREATE INDEX hippo_user IF NOT EXISTS FOR (n:HippoNode) ON (n.user_id)",
                    "CREATE INDEX hippo_created IF NOT EXISTS FOR (n:HippoNode) ON (n.created)",
                    "CREATE INDEX hippo_consolidated IF NOT EXISTS FOR (n:HippoNode) ON (n.consolidated)",
                ],
                "down": [
                    "DROP INDEX hyperedge_relation IF EXISTS",
                    "DROP INDEX hyperedge_confidence IF EXISTS",
                    "DROP INDEX hyperedge_timestamp IF EXISTS",
                    "DROP INDEX entity_type IF EXISTS",
                    "DROP INDEX hippo_session IF EXISTS",
                    "DROP INDEX hippo_user IF EXISTS",
                    "DROP INDEX hippo_created IF EXISTS",
                    "DROP INDEX hippo_consolidated IF EXISTS",
                ],
            },
            {
                "version": "003",
                "name": "create_hypergraph_schema",
                "description": "Create core hypergraph schema structure",
                "up": [
                    """
                    // Create hyperedge with metadata
                    CALL apoc.schema.assert(
                        {Hyperedge: ['id'], Entity: ['id'], HippoNode: ['id']},
                        {Hyperedge: [['relation', 'confidence']], Entity: [['type']], HippoNode: [['session_id', 'created']]}
                    )
                    """,
                    """
                    // Create relationship types for hyperedges
                    CREATE CONSTRAINT rel_type_constraint IF NOT EXISTS
                    FOR ()-[r:CONNECTS]-() REQUIRE r.hyperedge_id IS NOT NULL
                    """,
                ],
                "down": ["DROP CONSTRAINT rel_type_constraint IF EXISTS"],
            },
            {
                "version": "004",
                "name": "create_consolidation_procedures",
                "description": "Create stored procedures for memory consolidation",
                "up": [
                    """
                    // Procedure to find consolidation candidates
                    CREATE OR REPLACE PROCEDURE consolidation.findCandidates(threshold FLOAT)
                    YIELD node, score
                    CALL {
                        MATCH (h:HippoNode)
                        WHERE h.consolidated = false
                        AND h.consolidation_score >= threshold
                        RETURN h as node, h.consolidation_score as score
                    }
                    RETURN node, score
                    """,
                    """
                    // Procedure to consolidate hippo nodes to semantic graph
                    CREATE OR REPLACE PROCEDURE consolidation.consolidateNode(nodeId STRING)
                    YIELD success, message
                    CALL {
                        MATCH (h:HippoNode {id: nodeId})
                        SET h.consolidated = true, h.consolidation_timestamp = datetime()
                        RETURN h
                    }
                    RETURN true as success, 'Node consolidated' as message
                    """,
                ],
                "down": [
                    "DROP PROCEDURE consolidation.findCandidates IF EXISTS",
                    "DROP PROCEDURE consolidation.consolidateNode IF EXISTS",
                ],
            },
        ]

    def run_migrations(self, target_version: str | None = None) -> None:
        """Run migrations up to target version."""
        with self.driver.session() as session:
            # Check current version
            current_version = self._get_current_version(session)
            logger.info(f"Current schema version: {current_version}")

            # Run migrations
            for migration in self.migrations:
                if target_version and migration["version"] > target_version:
                    break

                if migration["version"] <= current_version:
                    continue

                logger.info(f"Running migration {migration['version']}: {migration['name']}")
                self._run_migration(session, migration)
                self._update_version(session, migration["version"])

    def rollback_migration(self, target_version: str) -> None:
        """Rollback to specific version."""
        with self.driver.session() as session:
            current_version = self._get_current_version(session)

            # Run rollbacks in reverse order
            for migration in reversed(self.migrations):
                if migration["version"] <= target_version:
                    break

                if migration["version"] > current_version:
                    continue

                logger.info(f"Rolling back migration {migration['version']}")
                self._rollback_migration(session, migration)

            self._update_version(session, target_version)

    def _get_current_version(self, session: Session) -> str:
        """Get current schema version."""
        try:
            result = session.run("MATCH (v:SchemaVersion) RETURN v.version ORDER BY v.version DESC LIMIT 1").single()
            return result["v.version"] if result else "000"
        except Exception:
            # Schema version tracking not yet created
            return "000"

    def _update_version(self, session: Session, version: str) -> None:
        """Update schema version."""
        session.run(
            "MERGE (v:SchemaVersion {version: $version}) "
            "ON CREATE SET v.created = datetime() "
            "ON MATCH SET v.updated = datetime()",
            version=version,
        )

    def _run_migration(self, session: Session, migration: dict[str, Any]) -> None:
        """Execute migration up queries."""
        for query in migration["up"]:
            try:
                session.run(query)
                logger.debug(f"Executed: {query[:100]}...")
            except Exception as e:
                logger.exception(f"Failed to execute migration query: {e}")
                logger.exception(f"Query: {query}")
                raise

    def _rollback_migration(self, session: Session, migration: dict[str, Any]) -> None:
        """Execute migration down queries."""
        for query in migration["down"]:
            try:
                session.run(query)
                logger.debug(f"Rolled back: {query[:100]}...")
            except Exception as e:
                logger.warning(f"Rollback query failed (may be expected): {e}")


# Convenience function for basic setup
def run_cypher_migrations(session: Session) -> None:
    """Run basic Cypher migrations for hypergraph schema.
    This is the function called by tests and setup scripts.
    """
    try:
        # Create basic constraints
        constraints = [
            "CREATE CONSTRAINT hyperedge_id IF NOT EXISTS FOR (h:Hyperedge) REQUIRE h.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT hippo_node_id IF NOT EXISTS FOR (n:HippoNode) REQUIRE n.id IS UNIQUE",
        ]

        # Create performance indexes
        indexes = [
            "CREATE INDEX hyperedge_relation IF NOT EXISTS FOR (h:Hyperedge) ON (h.relation)",
            "CREATE INDEX hyperedge_confidence IF NOT EXISTS FOR (h:Hyperedge) ON (h.confidence)",
            "CREATE INDEX hippo_session IF NOT EXISTS FOR (n:HippoNode) ON (n.session_id)",
            "CREATE INDEX hippo_consolidated IF NOT EXISTS FOR (n:HippoNode) ON (n.consolidated)",
        ]

        # Execute constraints
        for constraint in constraints:
            try:
                session.run(constraint)
                logger.debug(f"Created constraint: {constraint}")
            except Exception as e:
                logger.warning(f"Constraint creation failed (may already exist): {e}")

        # Execute indexes
        for index in indexes:
            try:
                session.run(index)
                logger.debug(f"Created index: {index}")
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")

        logger.info("Basic hypergraph schema migrations completed")

    except Exception as e:
        logger.exception(f"Migration failed: {e}")
        raise


# Async version for async drivers
async def run_cypher_migrations_async(session: AsyncSession) -> None:
    """Async version of migrations for AsyncSession."""
    try:
        constraints = [
            "CREATE CONSTRAINT hyperedge_id IF NOT EXISTS FOR (h:Hyperedge) REQUIRE h.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT hippo_node_id IF NOT EXISTS FOR (n:HippoNode) REQUIRE n.id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX hyperedge_relation IF NOT EXISTS FOR (h:Hyperedge) ON (h.relation)",
            "CREATE INDEX hyperedge_confidence IF NOT EXISTS FOR (h:Hyperedge) ON (h.confidence)",
            "CREATE INDEX hippo_session IF NOT EXISTS FOR (n:HippoNode) ON (n.session_id)",
            "CREATE INDEX hippo_consolidated IF NOT EXISTS FOR (n:HippoNode) ON (n.consolidated)",
        ]

        # Execute constraints
        for constraint in constraints:
            try:
                await session.run(constraint)
                logger.debug(f"Created constraint: {constraint}")
            except Exception as e:
                logger.warning(f"Constraint creation failed: {e}")

        # Execute indexes
        for index in indexes:
            try:
                await session.run(index)
                logger.debug(f"Created index: {index}")
            except Exception as e:
                logger.warning(f"Index creation failed: {e}")

        logger.info("Async hypergraph schema migrations completed")

    except Exception as e:
        logger.exception(f"Async migration failed: {e}")
        raise


# Database connection utilities
def create_neo4j_driver(
    uri: str = "bolt://localhost:7687",
    username: str = "neo4j",
    password: str = "aivillage_neo4j",
) -> Driver:
    """Create Neo4j driver with default settings."""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(
        uri,
        auth=(username, password),
        encrypted=False,  # For local development
    )


def verify_neo4j_connection(driver: Driver) -> bool:
    """Verify Neo4j connection is working."""
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            return result.single()["test"] == 1
    except Exception as e:
        logger.exception(f"Neo4j connection failed: {e}")
        return False


if __name__ == "__main__":
    # Allow running migrations directly
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        # Run migrations
        driver = create_neo4j_driver()
        if verify_neo4j_connection(driver):
            migrations = HypergraphMigrations(driver)
            migrations.run_migrations()
            print("Migrations completed successfully")
        else:
            print("Could not connect to Neo4j")
            sys.exit(1)
    else:
        print("Usage: python migrations.py run")
