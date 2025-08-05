"""GDC Extractor Engine.

Detects Graph Denial Constraint violations in Neo4j knowledge graphs.
"""

import asyncio
import logging
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import Neo4jError

from .registry import GDC_REGISTRY, get_enabled_gdcs
from .specs import GDCSpec, Violation

logger = logging.getLogger(__name__)


class GDCExtractor:
    """Graph Denial Constraint violation detection engine.

    Performs read-only analysis of Neo4j graphs to detect constraint violations.
    Supports both batch scanning and targeted GDC analysis.
    """

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_auth: tuple[str, str],
        max_concurrent_queries: int = 5,
        default_limit: int = 1000,
    ) -> None:
        """Initialize GDC extractor.

        Args:
            neo4j_uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            neo4j_auth: Tuple of (username, password)
            max_concurrent_queries: Maximum concurrent Cypher queries
            default_limit: Default LIMIT for Cypher queries
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = neo4j_auth
        self.max_concurrent_queries = max_concurrent_queries
        self.default_limit = default_limit

        self.driver: AsyncDriver | None = None
        self._semaphore = asyncio.Semaphore(max_concurrent_queries)

    async def initialize(self) -> None:
        """Initialize Neo4j driver connection."""
        try:
            self.driver = AsyncGraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)
            # Test connection
            async with self.driver.session() as session:
                await session.run("RETURN 1")
            logger.info(f"Connected to Neo4j: {self.neo4j_uri}")
        except Exception as e:
            logger.exception(f"Failed to connect to Neo4j: {e}")
            raise

    async def close(self) -> None:
        """Close Neo4j driver connection."""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j connection closed")

    async def scan_all(
        self,
        limit: int | None = None,
        enabled_only: bool = True,
        severity_filter: str | None = None,
    ) -> list[Violation]:
        """Scan for all GDC violations.

        Args:
            limit: Maximum violations per GDC (uses default_limit if None)
            enabled_only: Only scan enabled GDCs
            severity_filter: Only scan GDCs with specific severity

        Returns:
            List of detected violations
        """
        if not self.driver:
            msg = "Extractor not initialized. Call initialize() first."
            raise RuntimeError(msg)

        limit = limit or self.default_limit

        # Get GDCs to scan
        if enabled_only:
            gdcs_to_scan = get_enabled_gdcs(GDC_REGISTRY)
        else:
            gdcs_to_scan = list(GDC_REGISTRY.values())

        # Apply severity filter
        if severity_filter:
            gdcs_to_scan = [gdc for gdc in gdcs_to_scan if gdc.severity == severity_filter]

        if not gdcs_to_scan:
            logger.warning("No GDCs match scan criteria")
            return []

        logger.info(f"Scanning {len(gdcs_to_scan)} GDCs with limit {limit}")

        # Execute scans concurrently
        tasks = [self._scan_single_gdc(gdc, limit) for gdc in gdcs_to_scan]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Flatten results and handle exceptions
        all_violations = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to scan {gdcs_to_scan[i].id}: {result}")
            else:
                all_violations.extend(result)

        logger.info(f"Detected {len(all_violations)} total violations")
        return all_violations

    async def scan_gdc(self, gdc_id: str, limit: int | None = None) -> list[Violation]:
        """Scan for violations of a specific GDC.

        Args:
            gdc_id: GDC identifier to scan
            limit: Maximum violations to return

        Returns:
            List of detected violations for this GDC
        """
        if not self.driver:
            msg = "Extractor not initialized. Call initialize() first."
            raise RuntimeError(msg)

        if gdc_id not in GDC_REGISTRY:
            msg = f"Unknown GDC ID: {gdc_id}"
            raise ValueError(msg)

        gdc_spec = GDC_REGISTRY[gdc_id]
        limit = limit or self.default_limit

        logger.info(f"Scanning GDC {gdc_id} with limit {limit}")
        violations = await self._scan_single_gdc(gdc_spec, limit)
        logger.info(f"Detected {len(violations)} violations for {gdc_id}")

        return violations

    async def _scan_single_gdc(self, gdc_spec: GDCSpec, limit: int) -> list[Violation]:
        """Scan for violations of a single GDC."""
        async with self._semaphore:
            try:
                async with self.driver.session() as session:
                    return await self._execute_gdc_query(session, gdc_spec, limit)
            except Exception as e:
                logger.exception(f"Error scanning {gdc_spec.id}: {e}")
                return []

    async def _execute_gdc_query(self, session: AsyncSession, gdc_spec: GDCSpec, limit: int) -> list[Violation]:
        """Execute a GDC Cypher query and convert results to violations."""
        # Add LIMIT to query if not present
        cypher = gdc_spec.cypher.strip()
        if not cypher.lower().endswith(f"limit {limit}") and "limit" not in cypher.lower():
            cypher += f" LIMIT {limit}"

        try:
            result = await session.run(cypher)
            records = await result.data()

            violations = []
            for record in records:
                violation = await self._record_to_violation(record, gdc_spec)
                violations.append(violation)

            return violations

        except Neo4jError as e:
            logger.exception(f"Cypher error in {gdc_spec.id}: {e}")
            return []

    async def _record_to_violation(self, record: dict[str, Any], gdc_spec: GDCSpec) -> Violation:
        """Convert a Cypher query result record to a Violation object."""
        nodes = []
        edges = []
        relationships = []

        # Extract nodes, edges, and relationships from record
        for value in record.values():
            if hasattr(value, "labels"):  # Neo4j Node
                node_data = dict(value.items())
                node_data["_labels"] = list(value.labels)
                node_data["_neo4j_id"] = value.id
                nodes.append(node_data)

            elif hasattr(value, "type"):  # Neo4j Relationship
                rel_data = dict(value.items())
                rel_data["_type"] = value.type
                rel_data["_start_node_id"] = value.start_node.id
                rel_data["_end_node_id"] = value.end_node.id
                rel_data["_neo4j_id"] = value.id
                relationships.append(rel_data)

            elif isinstance(value, list):
                # Handle lists of nodes/relationships
                for item in value:
                    if hasattr(item, "labels"):
                        node_data = dict(item.items())
                        node_data["_labels"] = list(item.labels)
                        node_data["_neo4j_id"] = item.id
                        nodes.append(node_data)
                    elif hasattr(item, "type"):
                        rel_data = dict(item.items())
                        rel_data["_type"] = item.type
                        rel_data["_start_node_id"] = item.start_node.id
                        rel_data["_end_node_id"] = item.end_node.id
                        rel_data["_neo4j_id"] = item.id
                        relationships.append(rel_data)

        # Create violation object
        violation = Violation(
            gdc_id=gdc_spec.id,
            nodes=nodes,
            edges=edges,
            relationships=relationships,
            severity=gdc_spec.severity,
            suggested_repair=gdc_spec.suggested_action,
            metadata={
                "gdc_description": gdc_spec.description,
                "gdc_category": gdc_spec.category,
                "cypher_query": gdc_spec.cypher,
                "record_keys": list(record.keys()),
            },
        )

        return violation

    async def health_check(self) -> dict[str, Any]:
        """Check Neo4j connection health."""
        if not self.driver:
            return {"status": "disconnected", "error": "Driver not initialized"}

        try:
            async with self.driver.session() as session:
                result = await session.run("RETURN 1 as health")
                record = await result.single()

                if record and record["health"] == 1:
                    return {"status": "healthy", "neo4j_uri": self.neo4j_uri}
                return {"status": "unhealthy", "error": "Invalid response"}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def get_graph_stats(self) -> dict[str, Any]:
        """Get basic graph statistics for context."""
        if not self.driver:
            return {}

        try:
            async with self.driver.session() as session:
                # Get node counts by label
                node_result = await session.run(
                    """
                    CALL db.labels() YIELD label
                    CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {})
                    YIELD value
                    RETURN label, value.count as count
                """
                )
                node_counts = {record["label"]: record["count"] async for record in node_result}

                # Get relationship counts by type
                rel_result = await session.run(
                    """
                    CALL db.relationshipTypes() YIELD relationshipType
                    CALL apoc.cypher.run('MATCH ()-[r:' + relationshipType + ']->() RETURN count(r) as count', {})
                    YIELD value
                    RETURN relationshipType, value.count as count
                """
                )
                rel_counts = {record["relationshipType"]: record["count"] async for record in rel_result}

                return {
                    "node_counts": node_counts,
                    "relationship_counts": rel_counts,
                    "total_nodes": sum(node_counts.values()),
                    "total_relationships": sum(rel_counts.values()),
                }

        except Exception as e:
            logger.warning(f"Failed to get graph stats: {e}")
            return {"error": str(e)}


# Context manager for automatic connection handling
class GDCExtractorContext:
    """Context manager for GDCExtractor with automatic connection management."""

    def __init__(self, *args, **kwargs) -> None:
        self.extractor = GDCExtractor(*args, **kwargs)

    async def __aenter__(self) -> GDCExtractor:
        await self.extractor.initialize()
        return self.extractor

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.extractor.close()
