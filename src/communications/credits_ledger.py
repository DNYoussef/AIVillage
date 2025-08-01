"""Credits Ledger MVP - Fixed-supply shell currency with Prometheus-based earning."""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

logger = logging.getLogger(__name__)

Base = declarative_base()

# Database Models


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    node_id = Column(
        String(128), unique=True, nullable=True
    )  # For Prometheus node identification
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    wallet = relationship("Wallet", back_populates="user", uselist=False)
    sent_transactions = relationship(
        "Transaction",
        foreign_keys="Transaction.from_user_id",
        back_populates="from_user",
    )
    received_transactions = relationship(
        "Transaction", foreign_keys="Transaction.to_user_id", back_populates="to_user"
    )
    earnings = relationship("Earning", back_populates="user")


class Wallet(Base):
    __tablename__ = "wallets"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    balance = Column(Integer, default=0, nullable=False)  # Integer credits, no floats
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    user = relationship("User", back_populates="wallet")

    # Constraints
    __table_args__ = (Index("idx_wallet_user", "user_id"),)


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True)
    from_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    to_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    amount = Column(Integer, nullable=False)  # Amount before burn
    burn_amount = Column(Integer, nullable=False)  # 1% burn
    net_amount = Column(Integer, nullable=False)  # Amount after burn
    transaction_type = Column(String(32), nullable=False)  # 'transfer', 'earn', 'burn'
    status = Column(
        String(32), default="pending", nullable=False
    )  # 'pending', 'completed', 'failed'
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    from_user = relationship(
        "User", foreign_keys=[from_user_id], back_populates="sent_transactions"
    )
    to_user = relationship(
        "User", foreign_keys=[to_user_id], back_populates="received_transactions"
    )

    # Constraints
    __table_args__ = (
        Index("idx_transaction_from_user", "from_user_id"),
        Index("idx_transaction_to_user", "to_user_id"),
        Index("idx_transaction_created", "created_at"),
    )


class Earning(Base):
    __tablename__ = "earnings"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    scrape_timestamp = Column(DateTime, nullable=False)  # Prometheus scrape timestamp
    uptime_seconds = Column(Integer, nullable=False)
    flops = Column(Integer, nullable=False)
    bandwidth_bytes = Column(Integer, nullable=False)
    credits_earned = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="earnings")

    # Constraints - prevent double earning for same scrape
    __table_args__ = (
        UniqueConstraint("user_id", "scrape_timestamp", name="unique_user_scrape"),
        Index("idx_earning_user", "user_id"),
        Index("idx_earning_scrape", "scrape_timestamp"),
    )


# Data Transfer Objects


@dataclass
class BalanceResponse:
    user_id: int
    username: str
    balance: int
    last_updated: datetime


@dataclass
class TransactionResponse:
    id: int
    from_user: str
    to_user: str
    amount: int
    burn_amount: int
    net_amount: int
    transaction_type: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None


@dataclass
class EarningResponse:
    id: int
    user_id: int
    scrape_timestamp: datetime
    uptime_seconds: int
    flops: int
    bandwidth_bytes: int
    credits_earned: int
    created_at: datetime


# Configuration


class CreditsConfig:
    def __init__(self):
        self.database_url = os.getenv(
            "DATABASE_URL", "postgresql://user:password@localhost/aivillage"
        )
        self.burn_rate = 0.01  # 1% burn on spend
        self.fixed_supply = 1_000_000_000  # 1 billion credits max supply
        self.earning_rate_flops = 1000  # credits per GFLOP
        self.earning_rate_uptime = 10  # credits per hour
        self.earning_rate_bandwidth = 1  # credits per GB


# Core Ledger Logic


class CreditsLedger:
    def __init__(self, config: CreditsConfig):
        self.config = config
        self.engine = create_engine(config.database_url)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()

    def create_user(self, username: str, node_id: str | None = None) -> User:
        """Create a new user with wallet."""
        with self.get_session() as session:
            try:
                # Create user
                user = User(username=username, node_id=node_id)
                session.add(user)
                session.flush()  # Get user ID

                # Create wallet
                wallet = Wallet(user_id=user.id, balance=0)
                session.add(wallet)

                session.commit()
                return user
            except IntegrityError:
                session.rollback()
                raise ValueError(f"User {username} already exists")

    def get_user(self, username: str) -> User | None:
        """Get user by username."""
        with self.get_session() as session:
            return session.query(User).filter(User.username == username).first()

    def get_balance(self, username: str) -> BalanceResponse:
        """Get user balance."""
        with self.get_session() as session:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                raise ValueError(f"User {username} not found")

            wallet = user.wallet
            return BalanceResponse(
                user_id=user.id,
                username=user.username,
                balance=wallet.balance,
                last_updated=wallet.updated_at,
            )

    def transfer(
        self, from_username: str, to_username: str, amount: int
    ) -> TransactionResponse:
        """Transfer credits between users with 1% burn."""
        if amount <= 0:
            raise ValueError("Amount must be positive")

        burn_amount = int(amount * self.config.burn_rate)
        net_amount = amount - burn_amount

        with self.get_session() as session:
            # Get users
            from_user = (
                session.query(User).filter(User.username == from_username).first()
            )
            to_user = session.query(User).filter(User.username == to_username).first()

            if not from_user:
                raise ValueError(f"Sender {from_username} not found")
            if not to_user:
                raise ValueError(f"Recipient {to_username} not found")

            # Check balance
            if from_user.wallet.balance < amount:
                raise ValueError("Insufficient balance")

            # Create transaction record
            transaction = Transaction(
                from_user_id=from_user.id,
                to_user_id=to_user.id,
                amount=amount,
                burn_amount=burn_amount,
                net_amount=net_amount,
                transaction_type="transfer",
                status="pending",
            )
            session.add(transaction)

            try:
                # Update balances
                from_user.wallet.balance -= amount
                to_user.wallet.balance += net_amount

                # Complete transaction
                transaction.status = "completed"
                transaction.completed_at = datetime.now(timezone.utc)

                session.commit()

                return TransactionResponse(
                    id=transaction.id,
                    from_user=from_user.username,
                    to_user=to_user.username,
                    amount=amount,
                    burn_amount=burn_amount,
                    net_amount=net_amount,
                    transaction_type="transfer",
                    status="completed",
                    created_at=transaction.created_at,
                    completed_at=transaction.completed_at,
                )

            except Exception as e:
                session.rollback()
                transaction.status = "failed"
                session.commit()
                raise e

    def earn_credits(
        self,
        username: str,
        scrape_timestamp: datetime,
        uptime_seconds: int,
        flops: int,
        bandwidth_bytes: int,
    ) -> EarningResponse:
        """Mint credits based on Prometheus metrics. Idempotent by scrape timestamp."""
        with self.get_session() as session:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                raise ValueError(f"User {username} not found")

            # Check if already earned for this scrape timestamp
            existing_earning = (
                session.query(Earning)
                .filter(
                    Earning.user_id == user.id,
                    Earning.scrape_timestamp == scrape_timestamp,
                )
                .first()
            )

            if existing_earning:
                return EarningResponse(
                    id=existing_earning.id,
                    user_id=existing_earning.user_id,
                    scrape_timestamp=existing_earning.scrape_timestamp,
                    uptime_seconds=existing_earning.uptime_seconds,
                    flops=existing_earning.flops,
                    bandwidth_bytes=existing_earning.bandwidth_bytes,
                    credits_earned=existing_earning.credits_earned,
                    created_at=existing_earning.created_at,
                )

            # Calculate earnings
            uptime_hours = uptime_seconds / 3600
            flops_gflops = flops / 1e9
            bandwidth_gb = bandwidth_bytes / 1e9

            credits_earned = int(
                (uptime_hours * self.config.earning_rate_uptime)
                + (flops_gflops * self.config.earning_rate_flops)
                + (bandwidth_gb * self.config.earning_rate_bandwidth)
            )

            # Create earning record
            earning = Earning(
                user_id=user.id,
                scrape_timestamp=scrape_timestamp,
                uptime_seconds=uptime_seconds,
                flops=flops,
                bandwidth_bytes=bandwidth_bytes,
                credits_earned=credits_earned,
            )
            session.add(earning)

            # Update wallet balance
            user.wallet.balance += credits_earned

            session.commit()

            logger.info(f"User {username} earned {credits_earned} credits from metrics")

            return EarningResponse(
                id=earning.id,
                user_id=earning.user_id,
                scrape_timestamp=earning.scrape_timestamp,
                uptime_seconds=earning.uptime_seconds,
                flops=earning.flops,
                bandwidth_bytes=earning.bandwidth_bytes,
                credits_earned=earning.credits_earned,
                created_at=earning.created_at,
            )

    def get_transactions(
        self, username: str, limit: int = 100
    ) -> list[TransactionResponse]:
        """Get transaction history for user."""
        with self.get_session() as session:
            user = session.query(User).filter(User.username == username).first()
            if not user:
                raise ValueError(f"User {username} not found")

            # Get both sent and received transactions
            sent_txs = (
                session.query(Transaction)
                .filter(Transaction.from_user_id == user.id)
                .order_by(Transaction.created_at.desc())
                .limit(limit)
                .all()
            )

            received_txs = (
                session.query(Transaction)
                .filter(Transaction.to_user_id == user.id)
                .order_by(Transaction.created_at.desc())
                .limit(limit)
                .all()
            )

            # Combine and sort
            all_txs = sent_txs + received_txs
            all_txs.sort(key=lambda tx: tx.created_at, reverse=True)

            return [
                TransactionResponse(
                    id=tx.id,
                    from_user=tx.from_user.username,
                    to_user=tx.to_user.username,
                    amount=tx.amount,
                    burn_amount=tx.burn_amount,
                    net_amount=tx.net_amount,
                    transaction_type=tx.transaction_type,
                    status=tx.status,
                    created_at=tx.created_at,
                    completed_at=tx.completed_at,
                )
                for tx in all_txs[:limit]
            ]

    def get_total_supply(self) -> int:
        """Get total credits in circulation."""
        from sqlalchemy import func

        with self.get_session() as session:
            total = session.query(func.sum(Wallet.balance)).scalar() or 0
            return total
