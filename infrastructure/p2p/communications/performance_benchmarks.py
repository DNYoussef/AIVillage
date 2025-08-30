"""Performance benchmarks for credit system optimization validation."""

import asyncio
import time
import statistics
import logging
from datetime import datetime, UTC
from typing import List, Dict, Tuple
from dataclasses import dataclass
import concurrent.futures

from .credits_ledger import CreditsConfig, CreditsLedger, User

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Performance benchmark results."""
    operation: str
    total_time: float
    avg_time_per_operation: float
    operations_per_second: float
    min_time: float
    max_time: float
    p50_time: float
    p95_time: float
    p99_time: float
    total_operations: int
    errors: int
    success_rate: float


class CreditSystemBenchmark:
    """Performance benchmarking suite for credit system operations."""
    
    def __init__(self, database_url: str = "sqlite:///test_credits_benchmark.db"):
        """Initialize benchmark with test database."""
        self.config = CreditsConfig(database_url)
        self.ledger = CreditsLedger(self.config)
        self.ledger.create_tables()
        
    def setup_test_data(self, num_users: int = 1000) -> List[str]:
        """Create test users for benchmarking."""
        usernames = []
        
        start_time = time.time()
        for i in range(num_users):
            username = f"test_user_{i}"
            try:
                user = self.ledger.create_user(username, f"node_{i}")
                usernames.append(username)
                
                # Give users some initial balance for transfer tests
                if i % 2 == 0:  # Every other user gets credits
                    with self.ledger.get_session() as session:
                        user_obj = session.query(User).filter(User.username == username).first()
                        if user_obj and user_obj.wallet:
                            user_obj.wallet.balance = 1000
                            session.commit()
                            
            except ValueError:
                pass  # User already exists
                
        setup_time = time.time() - start_time
        logger.info(f"Created {len(usernames)} test users in {setup_time:.2f}s")
        return usernames
        
    def benchmark_single_balance_queries(self, usernames: List[str], iterations: int = 100) -> BenchmarkResult:
        """Benchmark individual balance queries (N+1 pattern - BEFORE optimization)."""
        times = []
        errors = 0
        
        test_usernames = usernames[:iterations]
        start_total = time.time()
        
        for username in test_usernames:
            start = time.time()
            try:
                self.ledger.get_balance(username)
                elapsed = time.time() - start
                times.append(elapsed)
            except Exception as e:
                errors += 1
                logger.error(f"Error getting balance for {username}: {e}")
                
        total_time = time.time() - start_total
        
        if times:
            return BenchmarkResult(
                operation="single_balance_queries",
                total_time=total_time,
                avg_time_per_operation=statistics.mean(times),
                operations_per_second=len(times) / total_time,
                min_time=min(times),
                max_time=max(times),
                p50_time=statistics.median(times),
                p95_time=self._percentile(times, 0.95),
                p99_time=self._percentile(times, 0.99),
                total_operations=len(test_usernames),
                errors=errors,
                success_rate=(len(times) / len(test_usernames)) * 100
            )
        else:
            return BenchmarkResult(
                operation="single_balance_queries",
                total_time=total_time,
                avg_time_per_operation=0,
                operations_per_second=0,
                min_time=0,
                max_time=0,
                p50_time=0,
                p95_time=0,
                p99_time=0,
                total_operations=len(test_usernames),
                errors=errors,
                success_rate=0
            )
            
    def benchmark_bulk_balance_queries(self, usernames: List[str], batch_size: int = 100) -> BenchmarkResult:
        """Benchmark bulk balance queries (AFTER optimization)."""
        times = []
        errors = 0
        
        start_total = time.time()
        
        # Process in batches
        for i in range(0, len(usernames), batch_size):
            batch = usernames[i:i + batch_size]
            start = time.time()
            try:
                results = self.ledger.bulk_get_balances(batch)
                elapsed = time.time() - start
                times.append(elapsed / len(batch))  # Time per operation
            except Exception as e:
                errors += 1
                logger.error(f"Error getting bulk balances: {e}")
                
        total_time = time.time() - start_total
        total_operations = len(usernames)
        
        if times:
            return BenchmarkResult(
                operation="bulk_balance_queries",
                total_time=total_time,
                avg_time_per_operation=statistics.mean(times),
                operations_per_second=total_operations / total_time,
                min_time=min(times),
                max_time=max(times),
                p50_time=statistics.median(times),
                p95_time=self._percentile(times, 0.95),
                p99_time=self._percentile(times, 0.99),
                total_operations=total_operations,
                errors=errors,
                success_rate=((len(times) * batch_size - errors) / total_operations) * 100
            )
        else:
            return BenchmarkResult(
                operation="bulk_balance_queries",
                total_time=total_time,
                avg_time_per_operation=0,
                operations_per_second=0,
                min_time=0,
                max_time=0,
                p50_time=0,
                p95_time=0,
                p99_time=0,
                total_operations=total_operations,
                errors=errors,
                success_rate=0
            )
            
    def benchmark_transaction_queries(self, usernames: List[str], limit: int = 100) -> BenchmarkResult:
        """Benchmark transaction history queries with JOIN optimization."""
        times = []
        errors = 0
        
        test_usernames = usernames[:min(50, len(usernames))]  # Limit for transaction tests
        start_total = time.time()
        
        for username in test_usernames:
            start = time.time()
            try:
                transactions = self.ledger.get_transactions(username, limit)
                elapsed = time.time() - start
                times.append(elapsed)
            except Exception as e:
                errors += 1
                logger.error(f"Error getting transactions for {username}: {e}")
                
        total_time = time.time() - start_total
        
        if times:
            return BenchmarkResult(
                operation="transaction_queries_optimized",
                total_time=total_time,
                avg_time_per_operation=statistics.mean(times),
                operations_per_second=len(times) / total_time,
                min_time=min(times),
                max_time=max(times),
                p50_time=statistics.median(times),
                p95_time=self._percentile(times, 0.95),
                p99_time=self._percentile(times, 0.99),
                total_operations=len(test_usernames),
                errors=errors,
                success_rate=(len(times) / len(test_usernames)) * 100
            )
        else:
            return BenchmarkResult(
                operation="transaction_queries_optimized",
                total_time=total_time,
                avg_time_per_operation=0,
                operations_per_second=0,
                min_time=0,
                max_time=0,
                p50_time=0,
                p95_time=0,
                p99_time=0,
                total_operations=len(test_usernames),
                errors=errors,
                success_rate=0
            )
            
    def benchmark_bulk_transfers(self, usernames: List[str], num_transfers: int = 100) -> BenchmarkResult:
        """Benchmark bulk transfer operations."""
        times = []
        errors = 0
        
        # Create transfer operations
        transfers = []
        for i in range(num_transfers):
            from_idx = i % len(usernames)
            to_idx = (i + 1) % len(usernames)
            transfers.append({
                'from_username': usernames[from_idx],
                'to_username': usernames[to_idx],
                'amount': 10
            })
            
        start_total = time.time()
        
        try:
            start = time.time()
            results = self.ledger.bulk_transfer(transfers)
            elapsed = time.time() - start
            times.append(elapsed / len(transfers))  # Time per transfer
        except Exception as e:
            errors += 1
            logger.error(f"Error in bulk transfer: {e}")
            
        total_time = time.time() - start_total
        
        if times:
            return BenchmarkResult(
                operation="bulk_transfers",
                total_time=total_time,
                avg_time_per_operation=statistics.mean(times),
                operations_per_second=num_transfers / total_time,
                min_time=min(times),
                max_time=max(times),
                p50_time=statistics.median(times),
                p95_time=self._percentile(times, 0.95),
                p99_time=self._percentile(times, 0.99),
                total_operations=num_transfers,
                errors=errors,
                success_rate=((num_transfers - errors) / num_transfers) * 100
            )
        else:
            return BenchmarkResult(
                operation="bulk_transfers",
                total_time=total_time,
                avg_time_per_operation=0,
                operations_per_second=0,
                min_time=0,
                max_time=0,
                p50_time=0,
                p95_time=0,
                p99_time=0,
                total_operations=num_transfers,
                errors=errors,
                success_rate=0
            )
    
    def run_comprehensive_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks and return results."""
        logger.info("Starting comprehensive credit system benchmark...")
        
        # Setup test data
        usernames = self.setup_test_data(1000)
        
        results = {}
        
        # Benchmark 1: Single balance queries (simulating N+1 problem)
        logger.info("Running single balance queries benchmark...")
        results['single_balance'] = self.benchmark_single_balance_queries(usernames, 200)
        
        # Benchmark 2: Bulk balance queries (optimization)
        logger.info("Running bulk balance queries benchmark...")
        results['bulk_balance'] = self.benchmark_bulk_balance_queries(usernames, 100)
        
        # Benchmark 3: Transaction queries with JOINs
        logger.info("Running transaction queries benchmark...")
        results['transaction_queries'] = self.benchmark_transaction_queries(usernames)
        
        # Benchmark 4: Bulk transfers
        logger.info("Running bulk transfers benchmark...")
        results['bulk_transfers'] = self.benchmark_bulk_transfers(usernames, 100)
        
        return results
        
    def calculate_performance_improvement(self, before: BenchmarkResult, after: BenchmarkResult) -> Dict[str, float]:
        """Calculate performance improvement percentage."""
        if before.avg_time_per_operation == 0:
            return {'improvement': 0}
            
        time_improvement = ((before.avg_time_per_operation - after.avg_time_per_operation) / before.avg_time_per_operation) * 100
        throughput_improvement = ((after.operations_per_second - before.operations_per_second) / before.operations_per_second) * 100
        
        return {
            'time_improvement_percent': time_improvement,
            'throughput_improvement_percent': throughput_improvement,
            'before_ops_per_sec': before.operations_per_second,
            'after_ops_per_sec': after.operations_per_second
        }
        
    def print_benchmark_report(self, results: Dict[str, BenchmarkResult]):
        """Print detailed benchmark report."""
        print("\n" + "="*80)
        print("CREDIT SYSTEM PERFORMANCE BENCHMARK REPORT")
        print("="*80)
        print(f"Timestamp: {datetime.now(UTC).isoformat()}")
        print(f"Database: {self.config.database_url}")
        print()
        
        for operation, result in results.items():
            print(f"Operation: {operation}")
            print(f"  Total Time: {result.total_time:.3f}s")
            print(f"  Average Time per Op: {result.avg_time_per_operation*1000:.2f}ms")
            print(f"  Operations/sec: {result.operations_per_second:.1f}")
            print(f"  P50 Latency: {result.p50_time*1000:.2f}ms")
            print(f"  P95 Latency: {result.p95_time*1000:.2f}ms")
            print(f"  P99 Latency: {result.p99_time*1000:.2f}ms")
            print(f"  Success Rate: {result.success_rate:.1f}%")
            print(f"  Total Operations: {result.total_operations}")
            print(f"  Errors: {result.errors}")
            print()
            
        # Calculate improvements
        if 'single_balance' in results and 'bulk_balance' in results:
            improvement = self.calculate_performance_improvement(
                results['single_balance'], 
                results['bulk_balance']
            )
            print("OPTIMIZATION IMPACT:")
            print(f"  Time Improvement: {improvement['time_improvement_percent']:.1f}%")
            print(f"  Throughput Improvement: {improvement['throughput_improvement_percent']:.1f}%")
            print(f"  Before: {improvement['before_ops_per_sec']:.1f} ops/sec")
            print(f"  After: {improvement['after_ops_per_sec']:.1f} ops/sec")
            
            if improvement['time_improvement_percent'] >= 80:
                print("  🎉 SUCCESS: Achieved 80%+ performance improvement target!")
            else:
                print(f"  ⚠️  WARNING: Only {improvement['time_improvement_percent']:.1f}% improvement (target: 80%+)")
                
        print("="*80)
        
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0
        data_sorted = sorted(data)
        index = int(len(data_sorted) * percentile)
        return data_sorted[min(index, len(data_sorted) - 1)]


def run_benchmark():
    """Main benchmark runner."""
    benchmark = CreditSystemBenchmark()
    
    try:
        results = benchmark.run_comprehensive_benchmark()
        benchmark.print_benchmark_report(results)
        
        return results
    finally:
        # Cleanup
        benchmark.ledger.engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_benchmark()