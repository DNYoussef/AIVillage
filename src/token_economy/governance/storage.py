"""Storage interface for governance state persistence."""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Protocol

from .models import Proposal

logger = logging.getLogger(__name__)


class GovernanceStorage(Protocol):
    """Interface for governance data persistence."""

    def save_proposal(self, proposal: Proposal) -> None:
        """Save or update a proposal."""
        ...

    def load_proposal(self, proposal_id: str) -> Proposal | None:
        """Load a proposal by ID."""
        ...

    def list_proposals(self) -> list[Proposal]:
        """List all proposals."""
        ...

    def delete_proposal(self, proposal_id: str) -> bool:
        """Delete a proposal."""
        ...

    def close(self) -> None:
        """Close storage connection."""
        ...


class SQLiteGovernanceStorage:
    """SQLite-based governance storage implementation."""

    def __init__(self, db_path: str) -> None:
        """Initialize SQLite storage."""
        self.path = Path(db_path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
        logger.info(f"Governance storage initialized at {self.path}")

    def _init_tables(self) -> None:
        """Create necessary database tables."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                proposer_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                voting_start INTEGER,
                voting_end INTEGER,
                execution_metadata TEXT,
                votes_json TEXT
            )
        """
        )
        self.conn.commit()
        logger.debug("Governance tables initialized")

    def save_proposal(self, proposal: Proposal) -> None:
        """Save or update a proposal."""
        proposal_data = proposal.to_dict()
        votes_json = json.dumps(proposal_data["votes"])
        execution_metadata_json = json.dumps(proposal_data["execution_metadata"])

        self.conn.execute(
            """
            INSERT OR REPLACE INTO proposals
            (id, title, description, proposer_id, status, created_at,
             voting_start, voting_end, execution_metadata, votes_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                proposal.id,
                proposal.title,
                proposal.description,
                proposal.proposer_id,
                proposal.status.value,
                proposal.created_at,
                proposal.voting_start,
                proposal.voting_end,
                execution_metadata_json,
                votes_json,
            ),
        )
        self.conn.commit()
        logger.debug(f"Proposal {proposal.id} saved")

    def load_proposal(self, proposal_id: str) -> Proposal | None:
        """Load a proposal by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM proposals WHERE id = ?", (proposal_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        # Reconstruct proposal data
        proposal_data = {
            "id": row["id"],
            "title": row["title"],
            "description": row["description"],
            "proposer_id": row["proposer_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "voting_start": row["voting_start"],
            "voting_end": row["voting_end"],
            "execution_metadata": json.loads(row["execution_metadata"] or "{}"),
            "votes": json.loads(row["votes_json"] or "[]"),
        }

        return Proposal.from_dict(proposal_data)

    def list_proposals(self) -> list[Proposal]:
        """List all proposals."""
        cursor = self.conn.execute("SELECT id FROM proposals ORDER BY created_at DESC")
        proposal_ids = [row["id"] for row in cursor.fetchall()]

        proposals = []
        for proposal_id in proposal_ids:
            proposal = self.load_proposal(proposal_id)
            if proposal:
                proposals.append(proposal)

        return proposals

    def delete_proposal(self, proposal_id: str) -> bool:
        """Delete a proposal."""
        cursor = self.conn.execute("DELETE FROM proposals WHERE id = ?", (proposal_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
        logger.debug("Governance storage connection closed")


class FileGovernanceStorage:
    """File-based governance storage implementation (JSON)."""

    def __init__(self, storage_dir: str = "governance_data") -> None:
        """Initialize file storage."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        logger.info(f"File governance storage initialized at {self.storage_dir}")

    def _get_proposal_path(self, proposal_id: str) -> Path:
        """Get file path for a proposal."""
        return self.storage_dir / f"proposal_{proposal_id}.json"

    def save_proposal(self, proposal: Proposal) -> None:
        """Save a proposal to file."""
        proposal_path = self._get_proposal_path(proposal.id)
        with proposal_path.open("w") as f:
            json.dump(proposal.to_dict(), f, indent=2)
        logger.debug(f"Proposal {proposal.id} saved to {proposal_path}")

    def load_proposal(self, proposal_id: str) -> Proposal | None:
        """Load a proposal from file."""
        proposal_path = self._get_proposal_path(proposal_id)
        if not proposal_path.exists():
            return None

        try:
            with proposal_path.open("r") as f:
                proposal_data = json.load(f)
            return Proposal.from_dict(proposal_data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading proposal {proposal_id}: {e}")
            return None

    def list_proposals(self) -> list[Proposal]:
        """List all proposals from files."""
        proposals = []
        for proposal_file in self.storage_dir.glob("proposal_*.json"):
            proposal_id = proposal_file.stem.replace("proposal_", "")
            proposal = self.load_proposal(proposal_id)
            if proposal:
                proposals.append(proposal)

        # Sort by creation time
        proposals.sort(key=lambda p: p.created_at, reverse=True)
        return proposals

    def delete_proposal(self, proposal_id: str) -> bool:
        """Delete a proposal file."""
        proposal_path = self._get_proposal_path(proposal_id)
        if proposal_path.exists():
            proposal_path.unlink()
            logger.debug(f"Proposal {proposal_id} deleted")
            return True
        return False

    def close(self) -> None:
        """No-op for file storage."""
