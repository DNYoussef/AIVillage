from dataclasses import dataclass
import json
import logging
from pathlib import Path
import sqlite3
import time

logger = logging.getLogger(__name__)

GLOBAL_SOUTH_COUNTRIES = {
    "Nigeria",
    "Kenya",
    "India",
    "Bangladesh",
    "Indonesia",
}

PPP_ADJUSTMENTS = {
    "Nigeria": 1.3,
    "Kenya": 1.2,
    "India": 1.1,
    "Bangladesh": 1.25,
    "Indonesia": 1.15,
}

QUALITY_THRESHOLD = 0.8


class SQLiteDatabase:
    """Lightweight wrapper around sqlite3 for thread-safe operations."""

    def __init__(self, db_path: str) -> None:
        """Initialise a new SQLite database connection."""
        self.path = Path(db_path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        logger.debug("SQLite database initialised at %s", self.path)

    def execute(self, query: str, params: tuple | None = None) -> sqlite3.Cursor:
        params = params or ()
        logger.debug("Executing SQL: %s | Params: %s", query, params)
        cur = self.conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.executescript(query)
        self.conn.commit()
        return cur

    def close(self) -> None:
        logger.debug("Closing SQLite connection")
        self.conn.close()


@dataclass
class EarningRule:
    action: str
    base_credits: int
    multipliers: dict[str, float]
    conditions: dict[str, str]


class VILLAGECreditSystem:
    """Off-chain credit system managing user balances and transactions."""

    def __init__(self, db_path: str = "village_credits.db") -> None:
        """Create the credit system using a SQLite backend."""
        self.db = SQLiteDatabase(db_path)
        self.init_tables()

    def init_tables(self) -> None:
        """Create necessary database tables."""
        self.db.execute(
            """
            CREATE TABLE IF NOT EXISTS balances (
                user_id TEXT PRIMARY KEY,
                balance INTEGER DEFAULT 0,
                earned_total INTEGER DEFAULT 0,
                spent_total INTEGER DEFAULT 0,
                last_updated INTEGER
            );

            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                amount INTEGER,
                tx_type TEXT,
                category TEXT,
                metadata TEXT,
                timestamp INTEGER
            );

            CREATE TABLE IF NOT EXISTS earning_rules (
                action TEXT PRIMARY KEY,
                base_credits INTEGER,
                multipliers TEXT,
                conditions TEXT
            );
            """
        )
        logger.info("Credit system tables ensured")

    def add_earning_rule(self, rule: EarningRule) -> None:
        """Insert or replace an earning rule."""
        self.db.execute(
            (
                "INSERT OR REPLACE INTO earning_rules "
                "(action, base_credits, multipliers, conditions) VALUES (?, ?, ?, ?)"
            ),
            (
                rule.action,
                rule.base_credits,
                json.dumps(rule.multipliers),
                json.dumps(rule.conditions),
            ),
        )
        logger.info("Earning rule for %s added", rule.action)

    def get_earning_rule(self, action: str) -> EarningRule:
        cur = self.db.execute(
            (
                "SELECT action, base_credits, multipliers, conditions "
                "FROM earning_rules WHERE action = ?"
            ),
            (action,),
        )
        row = cur.fetchone()
        if row is None:
            logger.error("Earning rule for action %s not found", action)
            message = f"Earning rule for action {action} not found"
            raise ValueError(message)
        return EarningRule(
            action=row["action"],
            base_credits=row["base_credits"],
            multipliers=json.loads(row["multipliers"] or "{}"),
            conditions=json.loads(row["conditions"] or "{}"),
        )

    def is_first_time(self, user_id: str, action: str) -> bool:
        cur = self.db.execute(
            (
                "SELECT COUNT(*) as cnt FROM transactions "
                "WHERE user_id = ? AND category = ?"
            ),
            (user_id, action),
        )
        count = cur.fetchone()["cnt"]
        logger.debug("User %s has %d prior %s actions", user_id, count, action)
        return count == 0

    def adjust_for_ppp(self, credit_amount: int, country: str | None) -> int:
        if country and country in PPP_ADJUSTMENTS:
            adjusted = int(credit_amount * PPP_ADJUSTMENTS[country])
            logger.debug(
                "PPP adjustment for %s: %d -> %d", country, credit_amount, adjusted
            )
            return adjusted
        return credit_amount

    def record_transaction(
        self,
        user_id: str,
        amount: int,
        tx_type: str,
        category: str,
        metadata: dict[str, str],
    ) -> None:
        tx_id = f"{user_id}-{int(time.time()*1000)}"
        self.db.execute(
            (
                "INSERT INTO transactions "
                "(id, user_id, amount, tx_type, category, metadata, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                tx_id,
                user_id,
                amount,
                tx_type,
                category,
                json.dumps(metadata),
                int(time.time()),
            ),
        )
        logger.info("Transaction %s recorded for user %s", tx_id, user_id)

    def update_balance(self, user_id: str, delta: int) -> None:
        cur = self.db.execute(
            "SELECT balance, earned_total, spent_total FROM balances WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if row is None:
            balance = max(delta, 0)
            earned_total = max(delta, 0)
            spent_total = max(-delta, 0)
            self.db.execute(
                (
                    "INSERT INTO balances (user_id, balance, earned_total, "
                    "spent_total, last_updated) VALUES (?, ?, ?, ?, ?)"
                ),
                (user_id, balance, earned_total, spent_total, int(time.time())),
            )
        else:
            balance = row["balance"] + delta
            earned_total = row["earned_total"] + max(delta, 0)
            spent_total = row["spent_total"] + max(-delta, 0)
            self.db.execute(
                (
                    "UPDATE balances SET balance = ?, earned_total = ?, "
                    "spent_total = ?, last_updated = ? WHERE user_id = ?"
                ),
                (balance, earned_total, spent_total, int(time.time()), user_id),
            )
        logger.debug("Balance updated for %s: %d", user_id, balance)

    def earn_credits(self, user_id: str, action: str, metadata: dict[str, str]) -> int:
        """Award credits for a specific action."""
        rule = self.get_earning_rule(action)
        credit_amount = rule.base_credits

        if metadata.get("location") in GLOBAL_SOUTH_COUNTRIES:
            credit_amount = int(credit_amount * 1.5)
        if metadata.get("quality_score", 0) > QUALITY_THRESHOLD:
            credit_amount = int(credit_amount * 1.2)
        if self.is_first_time(user_id, action):
            credit_amount *= 2
        credit_amount = self.adjust_for_ppp(credit_amount, metadata.get("country"))

        self.record_transaction(
            user_id=user_id,
            amount=credit_amount,
            tx_type="EARN",
            category=action,
            metadata=metadata,
        )
        self.update_balance(user_id, credit_amount)
        return credit_amount

    def spend_credits(
        self, user_id: str, amount: int, category: str, metadata: dict[str, str]
    ) -> None:
        cur = self.db.execute(
            "SELECT balance FROM balances WHERE user_id = ?", (user_id,)
        )
        row = cur.fetchone()
        if row is None or row["balance"] < amount:
            logger.error("User %s has insufficient balance", user_id)
            message = "Insufficient balance"
            raise ValueError(message)
        self.record_transaction(user_id, -amount, "SPEND", category, metadata)
        self.update_balance(user_id, -amount)

    def get_balance(self, user_id: str) -> int:
        cur = self.db.execute(
            "SELECT balance FROM balances WHERE user_id = ?", (user_id,)
        )
        row = cur.fetchone()
        return row["balance"] if row else 0

    def close(self) -> None:
        self.db.close()
