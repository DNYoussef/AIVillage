"""
On-Device Mini-RAG System for Edge Computing

A lightweight, privacy-focused RAG system that runs entirely on the user's device:
- Personal knowledge base for individual user context
- Privacy-preserving knowledge elevation to global RAG
- Local vector embeddings and semantic search
- Automatic knowledge relevance scoring
- Non-identifying information extraction for global contributions

This system allows digital twins to maintain personal knowledge while contributing
anonymized insights to the global knowledge base managed by distributed systems.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import time
from typing import Any

import numpy as np

try:
    from ..integration.shared_types import DataSource, PrivacyLevel
except ImportError:
    try:
        from packages.edge.mobile.shared_types import DataSource, PrivacyLevel
    except ImportError:
        # Define minimal types for standalone operation
        from enum import Enum

        class DataSource(Enum):
            CONVERSATION = "conversation"
            PURCHASE = "purchase"
            LOCATION = "location"
            APP_USAGE = "app_usage"
            CALENDAR = "calendar"
            VOICE = "voice"

        class PrivacyLevel(Enum):
            PUBLIC = "public"
            PRIVATE = "private"
            PERSONAL = "personal"
            SENSITIVE = "sensitive"


logger = logging.getLogger(__name__)


class KnowledgeRelevance(Enum):
    """Relevance levels for knowledge pieces"""

    PERSONAL_ONLY = "personal_only"  # Never share, personal context only
    LOCAL_INSIGHT = "local_insight"  # Might be useful for similar users
    GENERAL_PATTERN = "general_pattern"  # General behavioral/usage patterns
    GLOBAL_KNOWLEDGE = "global_knowledge"  # Globally important information
    CRITICAL_DISCOVERY = "critical_discovery"  # Urgent global importance


@dataclass
class KnowledgePiece:
    """Individual piece of knowledge in the mini-RAG system"""

    knowledge_id: str
    content: str  # The actual knowledge content
    source: DataSource  # Where this knowledge came from
    privacy_level: PrivacyLevel
    relevance: KnowledgeRelevance

    # Embeddings and search
    embedding: np.ndarray | None = None
    keywords: list[str] = field(default_factory=list)

    # Context and metadata
    context: dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.5
    usage_frequency: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)

    # Global contribution tracking
    contributed_to_global: bool = False
    contribution_hash: str | None = None
    anonymization_level: float = 0.0  # How much anonymization was applied

    def anonymize_for_global_sharing(self) -> dict[str, Any]:
        """Create anonymized version for global RAG contribution"""

        # Remove personal identifiers
        anonymized_content = self._remove_personal_identifiers(self.content)

        # Create pattern-based knowledge instead of specific instances
        pattern_knowledge = self._extract_behavioral_pattern(anonymized_content)

        # Generate contribution hash for deduplication
        content_hash = hashlib.sha256(pattern_knowledge.encode()).hexdigest()

        return {
            "pattern_knowledge": pattern_knowledge,
            "knowledge_type": self.source.value,
            "relevance_score": self.confidence_score,
            "usage_frequency": min(self.usage_frequency, 100),  # Cap to prevent identification
            "contribution_hash": content_hash,
            "anonymization_applied": True,
            "privacy_preserved": True,
            "global_relevance": self.relevance.value,
            "timestamp_range": self._get_time_bucket(),  # Week/month bucket instead of exact time
            "pattern_confidence": self.confidence_score,
        }

    def _remove_personal_identifiers(self, content: str) -> str:
        """Remove names, locations, specific times, etc."""
        import re

        # Replace specific patterns with generic ones
        anonymized = content

        # Replace names with generic references
        anonymized = re.sub(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", "[person]", anonymized)

        # Replace specific locations with generic ones
        anonymized = re.sub(
            r"\b\d{1,5}\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd)\b", "[address]", anonymized
        )

        # Replace phone numbers, emails
        anonymized = re.sub(r"\b\d{3}-?\d{3}-?\d{4}\b", "[phone]", anonymized)
        anonymized = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[email]", anonymized)

        # Replace specific times with time ranges
        anonymized = re.sub(r"\b\d{1,2}:\d{2}\s?[AP]M\b", "[time]", anonymized)

        return anonymized

    def _extract_behavioral_pattern(self, anonymized_content: str) -> str:
        """Extract behavioral patterns rather than specific instances"""

        # Convert specific events to general patterns
        patterns = {
            "frequently uses app": "high_app_engagement_pattern",
            "visits location": "location_visit_pattern",
            "purchases item": "purchase_behavior_pattern",
            "communicates with": "communication_pattern",
            "schedules meeting": "scheduling_pattern",
        }

        pattern_content = anonymized_content
        for specific, pattern in patterns.items():
            if specific in anonymized_content.lower():
                pattern_content = f"{pattern}: {self.source.value}"
                break

        return pattern_content

    def _get_time_bucket(self) -> str:
        """Get time bucket instead of exact timestamp"""
        now = datetime.now()

        # Use week buckets to preserve some temporal info while anonymizing
        year = now.year
        week = now.isocalendar()[1]

        return f"{year}-W{week:02d}"


@dataclass
class GlobalContribution:
    """Tracks contributions to global RAG system"""

    contribution_id: str
    original_knowledge_id: str
    anonymized_content: dict[str, Any]
    contribution_timestamp: datetime
    accepted_by_global: bool = False
    global_integration_status: str = "pending"


class MiniRAGSystem:
    """
    Lightweight RAG system running entirely on device for edge computing

    Features:
    - Personal knowledge base with vector search
    - Privacy-preserving knowledge elevation
    - Local semantic embeddings
    - Auto-contribution to global RAG (anonymized)
    - Integration with digital twin systems
    """

    def __init__(self, data_dir: Path, twin_id: str):
        self.data_dir = data_dir
        self.twin_id = twin_id
        self.db_path = data_dir / f"mini_rag_{twin_id}.db"

        # Knowledge storage
        self.knowledge_base: dict[str, KnowledgePiece] = {}
        self.pending_contributions: list[GlobalContribution] = []

        # Simple embedding system (in production would use sentence-transformers)
        self.embedding_dim = 384

        self._setup_database()
        self._load_existing_knowledge()

        logger.info(f"Mini-RAG system initialized for twin {twin_id}")

    def _setup_database(self):
        """Initialize SQLite database for knowledge storage"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            # Knowledge pieces table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_pieces (
                    knowledge_id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    privacy_level INTEGER NOT NULL,
                    relevance TEXT NOT NULL,
                    embedding BLOB,
                    keywords TEXT,  -- JSON array
                    context TEXT,   -- JSON object
                    confidence_score REAL DEFAULT 0.5,
                    usage_frequency INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    created_at TEXT,
                    contributed_to_global BOOLEAN DEFAULT FALSE,
                    contribution_hash TEXT,
                    anonymization_level REAL DEFAULT 0.0
                )
            """
            )

            # Global contributions tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS global_contributions (
                    contribution_id TEXT PRIMARY KEY,
                    original_knowledge_id TEXT,
                    anonymized_content TEXT,  -- JSON
                    contribution_timestamp TEXT,
                    accepted_by_global BOOLEAN DEFAULT FALSE,
                    global_integration_status TEXT DEFAULT 'pending'
                )
            """
            )

            # Indexes for search performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relevance ON knowledge_pieces(relevance)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON knowledge_pieces(source)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_privacy_level ON knowledge_pieces(privacy_level)")

    def _load_existing_knowledge(self):
        """Load existing knowledge from database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT knowledge_id, content, source, privacy_level, relevance,
                       embedding, keywords, context, confidence_score, usage_frequency,
                       last_accessed, created_at, contributed_to_global,
                       contribution_hash, anonymization_level
                FROM knowledge_pieces
            """
            )

            for row in cursor.fetchall():
                (
                    knowledge_id,
                    content,
                    source,
                    privacy_level,
                    relevance,
                    embedding_bytes,
                    keywords_json,
                    context_json,
                    confidence_score,
                    usage_frequency,
                    last_accessed,
                    created_at,
                    contributed_to_global,
                    contribution_hash,
                    anonymization_level,
                ) = row

                # Deserialize data
                embedding = np.frombuffer(embedding_bytes) if embedding_bytes else None
                keywords = json.loads(keywords_json) if keywords_json else []
                context = json.loads(context_json) if context_json else {}

                knowledge = KnowledgePiece(
                    knowledge_id=knowledge_id,
                    content=content,
                    source=DataSource(source),
                    privacy_level=PrivacyLevel(privacy_level),
                    relevance=KnowledgeRelevance(relevance),
                    embedding=embedding,
                    keywords=keywords,
                    context=context,
                    confidence_score=confidence_score,
                    usage_frequency=usage_frequency,
                    last_accessed=datetime.fromisoformat(last_accessed),
                    created_at=datetime.fromisoformat(created_at),
                    contributed_to_global=bool(contributed_to_global),
                    contribution_hash=contribution_hash,
                    anonymization_level=anonymization_level,
                )

                self.knowledge_base[knowledge_id] = knowledge

        logger.info(f"Loaded {len(self.knowledge_base)} knowledge pieces")

    async def add_knowledge(self, content: str, source: DataSource, context: dict[str, Any] = None) -> str:
        """Add new knowledge to the personal RAG system"""

        knowledge_id = f"knowledge_{int(time.time())}_{hash(content) % 10000}"

        # Determine privacy level and relevance
        privacy_level = self._assess_privacy_level(content, context or {})
        relevance = self._assess_global_relevance(content, source, context or {})

        # Generate simple embedding (in production would use proper embeddings)
        embedding = self._generate_embedding(content)

        # Extract keywords
        keywords = self._extract_keywords(content)

        knowledge = KnowledgePiece(
            knowledge_id=knowledge_id,
            content=content,
            source=source,
            privacy_level=privacy_level,
            relevance=relevance,
            embedding=embedding,
            keywords=keywords,
            context=context or {},
            confidence_score=0.7,  # Default confidence
            usage_frequency=1,
        )

        # Store in memory and database
        self.knowledge_base[knowledge_id] = knowledge
        await self._persist_knowledge(knowledge)

        # Check if this should be contributed to global RAG
        if relevance in [KnowledgeRelevance.GLOBAL_KNOWLEDGE, KnowledgeRelevance.CRITICAL_DISCOVERY]:
            await self._queue_for_global_contribution(knowledge)

        logger.info(f"Added knowledge piece {knowledge_id} with relevance {relevance.value}")
        return knowledge_id

    async def query_knowledge(self, query: str, max_results: int = 5) -> list[KnowledgePiece]:
        """Query the personal knowledge base"""

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Calculate similarity scores
        results = []
        for knowledge in self.knowledge_base.values():
            if knowledge.embedding is not None:
                similarity = self._cosine_similarity(query_embedding, knowledge.embedding)
                results.append((similarity, knowledge))

        # Sort by similarity and return top results
        results.sort(key=lambda x: x[0], reverse=True)
        top_results = [knowledge for _, knowledge in results[:max_results]]

        # Update usage frequency
        for knowledge in top_results:
            knowledge.usage_frequency += 1
            knowledge.last_accessed = datetime.now()
            await self._persist_knowledge(knowledge)

        return top_results

    async def get_global_contribution_candidates(self) -> list[KnowledgePiece]:
        """Get knowledge pieces that could be contributed to global RAG"""

        candidates = []
        for knowledge in self.knowledge_base.values():
            if (
                not knowledge.contributed_to_global
                and knowledge.relevance
                in [
                    KnowledgeRelevance.GENERAL_PATTERN,
                    KnowledgeRelevance.GLOBAL_KNOWLEDGE,
                    KnowledgeRelevance.CRITICAL_DISCOVERY,
                ]
                and knowledge.usage_frequency >= 2
            ):  # Must be used multiple times
                candidates.append(knowledge)

        # Sort by relevance and confidence
        candidates.sort(
            key=lambda k: (
                k.relevance == KnowledgeRelevance.CRITICAL_DISCOVERY,
                k.relevance == KnowledgeRelevance.GLOBAL_KNOWLEDGE,
                k.confidence_score,
            ),
            reverse=True,
        )

        return candidates

    async def contribute_to_global_rag(self, knowledge_ids: list[str]) -> list[GlobalContribution]:
        """Prepare anonymized knowledge for global RAG contribution"""

        contributions = []

        for knowledge_id in knowledge_ids:
            if knowledge_id not in self.knowledge_base:
                continue

            knowledge = self.knowledge_base[knowledge_id]

            # Skip if already contributed or too personal
            if knowledge.contributed_to_global or knowledge.privacy_level == PrivacyLevel.CONFIDENTIAL:
                continue

            # Create anonymized version
            anonymized = knowledge.anonymize_for_global_sharing()

            contribution = GlobalContribution(
                contribution_id=f"contrib_{int(time.time())}_{knowledge_id}",
                original_knowledge_id=knowledge_id,
                anonymized_content=anonymized,
                contribution_timestamp=datetime.now(),
            )

            contributions.append(contribution)
            self.pending_contributions.append(contribution)

            # Mark as contributed
            knowledge.contributed_to_global = True
            knowledge.contribution_hash = anonymized["contribution_hash"]
            await self._persist_knowledge(knowledge)

        logger.info(f"Created {len(contributions)} global contributions")
        return contributions

    def _assess_privacy_level(self, content: str, context: dict) -> PrivacyLevel:
        """Assess privacy level of content"""

        # Simple privacy assessment (in production would be more sophisticated)
        personal_indicators = ["I", "my", "me", "personal", "private", "home", "family"]
        sensitive_indicators = ["password", "SSN", "credit card", "bank", "medical"]

        content_lower = content.lower()

        if any(indicator in content_lower for indicator in sensitive_indicators):
            return PrivacyLevel.CONFIDENTIAL
        elif any(indicator in content_lower for indicator in personal_indicators):
            return PrivacyLevel.PERSONAL
        else:
            return PrivacyLevel.PUBLIC

    def _assess_global_relevance(self, content: str, source: DataSource, context: dict) -> KnowledgeRelevance:
        """Assess whether this knowledge could be globally relevant"""

        content_lower = content.lower()

        # Critical discoveries
        critical_indicators = ["security vulnerability", "major bug", "critical issue", "urgent", "emergency"]
        if any(indicator in content_lower for indicator in critical_indicators):
            return KnowledgeRelevance.CRITICAL_DISCOVERY

        # Global knowledge indicators
        global_indicators = ["how to", "tutorial", "guide", "best practice", "optimization", "performance"]
        if any(indicator in content_lower for indicator in global_indicators):
            return KnowledgeRelevance.GLOBAL_KNOWLEDGE

        # General patterns
        pattern_indicators = ["pattern", "trend", "behavior", "usage", "frequently", "often"]
        if any(indicator in content_lower for indicator in pattern_indicators):
            return KnowledgeRelevance.GENERAL_PATTERN

        # Local insights based on usage frequency context
        if context.get("usage_frequency", 0) > 5:
            return KnowledgeRelevance.LOCAL_INSIGHT

        return KnowledgeRelevance.PERSONAL_ONLY

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate simple embedding for text (placeholder for real embeddings)"""
        # Simple bag-of-words style embedding for demo
        # In production would use sentence-transformers or similar

        words = text.lower().split()
        embedding = np.zeros(self.embedding_dim)

        for i, word in enumerate(words[: self.embedding_dim]):
            # Simple hash-based embedding
            hash_val = hash(word) % self.embedding_dim
            embedding[hash_val] += 1.0 / (i + 1)  # Position weighting

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _extract_keywords(self, content: str) -> list[str]:
        """Extract keywords from content"""
        import re

        # Simple keyword extraction
        words = re.findall(r"\b\w+\b", content.lower())

        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # Return unique keywords, limited to top 10
        return list(set(keywords))[:10]

    async def _persist_knowledge(self, knowledge: KnowledgePiece):
        """Persist knowledge piece to database"""

        # Serialize data
        embedding_bytes = knowledge.embedding.tobytes() if knowledge.embedding is not None else None
        keywords_json = json.dumps(knowledge.keywords)
        context_json = json.dumps(knowledge.context)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO knowledge_pieces
                (knowledge_id, content, source, privacy_level, relevance, embedding,
                 keywords, context, confidence_score, usage_frequency, last_accessed,
                 created_at, contributed_to_global, contribution_hash, anonymization_level)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    knowledge.knowledge_id,
                    knowledge.content,
                    knowledge.source.value,
                    knowledge.privacy_level.value,
                    knowledge.relevance.value,
                    embedding_bytes,
                    keywords_json,
                    context_json,
                    knowledge.confidence_score,
                    knowledge.usage_frequency,
                    knowledge.last_accessed.isoformat(),
                    knowledge.created_at.isoformat(),
                    knowledge.contributed_to_global,
                    knowledge.contribution_hash,
                    knowledge.anonymization_level,
                ),
            )

    async def _queue_for_global_contribution(self, knowledge: KnowledgePiece):
        """Queue knowledge for potential global RAG contribution"""

        # Only queue if meets criteria and isn't too personal
        if knowledge.privacy_level in [PrivacyLevel.PUBLIC, PrivacyLevel.PERSONAL] and knowledge.relevance in [
            KnowledgeRelevance.GLOBAL_KNOWLEDGE,
            KnowledgeRelevance.CRITICAL_DISCOVERY,
        ]:
            logger.info(f"Queued knowledge {knowledge.knowledge_id} for global contribution")
            # Would trigger async contribution process here

    async def cleanup_old_knowledge(self, retention_days: int = 30):
        """Clean up old, unused knowledge"""

        cutoff_time = datetime.now() - timedelta(days=retention_days)

        to_remove = []
        for knowledge_id, knowledge in self.knowledge_base.items():
            if (
                knowledge.last_accessed < cutoff_time
                and knowledge.usage_frequency < 2
                and not knowledge.contributed_to_global
            ):
                to_remove.append(knowledge_id)

        # Remove from memory and database
        for knowledge_id in to_remove:
            del self.knowledge_base[knowledge_id]

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM knowledge_pieces WHERE knowledge_id = ?", (knowledge_id,))

        logger.info(f"Cleaned up {len(to_remove)} old knowledge pieces")

    def get_system_stats(self) -> dict[str, Any]:
        """Get statistics about the mini-RAG system"""

        stats = {
            "total_knowledge_pieces": len(self.knowledge_base),
            "by_relevance": {},
            "by_source": {},
            "by_privacy_level": {},
            "contributions_pending": len(self.pending_contributions),
            "contributed_to_global": sum(1 for k in self.knowledge_base.values() if k.contributed_to_global),
            "average_usage_frequency": np.mean([k.usage_frequency for k in self.knowledge_base.values()])
            if self.knowledge_base
            else 0,
        }

        # Count by categories
        for knowledge in self.knowledge_base.values():
            relevance = knowledge.relevance.value
            source = knowledge.source.value
            privacy = knowledge.privacy_level.name

            stats["by_relevance"][relevance] = stats["by_relevance"].get(relevance, 0) + 1
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1
            stats["by_privacy_level"][privacy] = stats["by_privacy_level"].get(privacy, 0) + 1

        return stats


# Integration example
async def demo_mini_rag_system():
    """Demonstrate the mini-RAG system"""
    print("🧠 Mini-RAG System Demo")
    print("=" * 50)

    # Create mini-RAG system
    data_dir = Path("./mini_rag_data")
    rag = MiniRAGSystem(data_dir, "demo_twin")

    # Add some knowledge
    print("\n📚 Adding knowledge...")
    await rag.add_knowledge(
        "User frequently uses productivity apps during work hours (9-5)",
        DataSource.APP_USAGE,
        {"pattern": "work_productivity", "confidence": 0.8},
    )

    await rag.add_knowledge(
        "Best practice: Use dark mode to reduce battery consumption on OLED screens",
        DataSource.APP_USAGE,
        {"type": "optimization", "global_relevance": True},
    )

    await rag.add_knowledge(
        "Personal reminder: Take medication at 8 AM daily", DataSource.CALENDAR, {"personal": True, "recurring": True}
    )

    # Query knowledge
    print("\n🔍 Querying knowledge...")
    results = await rag.query_knowledge("productivity apps")
    for result in results:
        print(f"  Found: {result.content[:100]}... (relevance: {result.relevance.value})")

    # Check global contribution candidates
    print("\n🌍 Global contribution candidates...")
    candidates = await rag.get_global_contribution_candidates()
    for candidate in candidates:
        print(f"  Candidate: {candidate.content[:100]}... (relevance: {candidate.relevance.value})")

    # Create contributions
    if candidates:
        contributions = await rag.contribute_to_global_rag([c.knowledge_id for c in candidates])
        print(f"\n📤 Created {len(contributions)} anonymized contributions")
        for contrib in contributions:
            print(f"  Contribution: {contrib.anonymized_content['pattern_knowledge']}")

    # Show stats
    print("\n📊 System Stats:")
    stats = rag.get_system_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_mini_rag_system())