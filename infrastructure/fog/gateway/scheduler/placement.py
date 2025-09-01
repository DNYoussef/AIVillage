"""
NSGA-II Multi-Objective Fog Job Scheduler

Implements Non-dominated Sorting Genetic Algorithm II (NSGA-II) for optimal
fog job placement across distributed nodes. Optimizes multiple objectives:
- Minimize execution latency
- Balance computational load
- Maximize node trust scores
- Minimize operational costs

This scheduler integrates with the existing fog gateway infrastructure
to provide intelligent job placement with Pareto-optimal solutions.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
import logging
import random
import time
from typing import Any

# Import marketplace for pricing constraints
from .marketplace import BidType, PricingTier

logger = logging.getLogger(__name__)


class PlacementStrategy(str, Enum):
    """Job placement strategy options"""

    NSGA_II = "nsga_ii"  # Multi-objective optimization (default)
    LATENCY_FIRST = "latency"  # Minimize latency only
    LOAD_BALANCE = "load_balance"  # Balance load only
    TRUST_FIRST = "trust"  # Maximize trust only
    COST_OPTIMIZE = "cost"  # Minimize cost only
    ROUND_ROBIN = "round_robin"  # Simple round-robin


class JobClass(str, Enum):
    """SLA-based job classification"""

    S_CLASS = "S"  # replicated + attested
    A_CLASS = "A"  # replicated
    B_CLASS = "B"  # best-effort


@dataclass
class FogNode:
    """Fog computing node representation"""

    node_id: str
    endpoint: str

    # Resource capacity
    cpu_cores: float = 8.0
    memory_gb: float = 16.0
    disk_gb: float = 100.0

    # Current utilization (0.0-1.0)
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    disk_utilization: float = 0.0

    # Performance metrics
    avg_latency_ms: float = 100.0
    success_rate: float = 1.0
    trust_score: float = 0.8

    # Cost model
    cpu_cost_per_hour: float = 0.10
    memory_cost_per_gb_hour: float = 0.05

    # Network properties
    region: str = "default"
    availability_zone: str = "default"
    network_bandwidth_mbps: float = 1000.0

    # Current job load
    active_jobs: int = 0
    queued_jobs: int = 0

    # Node status
    is_healthy: bool = True
    last_heartbeat: datetime | None = None

    def available_cpu(self) -> float:
        """Get available CPU cores"""
        return self.cpu_cores * (1.0 - self.cpu_utilization)

    def available_memory(self) -> float:
        """Get available memory in GB"""
        return self.memory_gb * (1.0 - self.memory_utilization)

    def available_disk(self) -> float:
        """Get available disk space in GB"""
        return self.disk_gb * (1.0 - self.disk_utilization)

    def can_handle_job(self, job_requirements: dict[str, float]) -> bool:
        """Check if node can handle job resource requirements"""
        required_cpu = job_requirements.get("cpu_cores", 1.0)
        required_memory = job_requirements.get("memory_gb", 1.0)
        required_disk = job_requirements.get("disk_gb", 1.0)

        return (
            self.is_healthy
            and self.available_cpu() >= required_cpu
            and self.available_memory() >= required_memory
            and self.available_disk() >= required_disk
        )

    def estimate_latency(self, job_size_mb: float) -> float:
        """Estimate job execution latency"""
        # Base latency + transfer time + queue delay
        transfer_time = job_size_mb / (self.network_bandwidth_mbps / 8)
        queue_delay = self.queued_jobs * 10.0  # 10ms per queued job
        load_penalty = self.cpu_utilization * 50.0  # Load-based penalty

        return self.avg_latency_ms + transfer_time + queue_delay + load_penalty

    def estimate_cost(self, job_requirements: dict[str, float], duration_hours: float) -> float:
        """Estimate job execution cost"""
        cpu_cost = job_requirements.get("cpu_cores", 1.0) * self.cpu_cost_per_hour * duration_hours
        memory_cost = job_requirements.get("memory_gb", 1.0) * self.memory_cost_per_gb_hour * duration_hours

        return cpu_cost + memory_cost


@dataclass
class JobRequest:
    """Fog job request with requirements and constraints"""

    job_id: str
    namespace: str
    job_class: JobClass = JobClass.B_CLASS

    # Resource requirements
    cpu_cores: float = 1.0
    memory_gb: float = 1.0
    disk_gb: float = 1.0

    # Job characteristics
    estimated_duration_hours: float = 1.0
    job_size_mb: float = 10.0
    priority: int = 5  # 1-10, 10 = highest

    # Constraints
    region_preference: str | None = None
    max_latency_ms: float | None = None
    max_cost: float | None = None
    excluded_nodes: set[str] = field(default_factory=set)

    # Marketplace constraints
    max_price: float | None = None  # Maximum willing to pay in USD
    bid_type: BidType = BidType.SPOT
    pricing_tier: PricingTier = PricingTier.BASIC
    min_trust_score: float = 0.3

    # Timestamps
    submitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    deadline: datetime | None = None

    def to_requirements_dict(self) -> dict[str, float]:
        """Convert to resource requirements dictionary"""
        return {"cpu_cores": self.cpu_cores, "memory_gb": self.memory_gb, "disk_gb": self.disk_gb}


@dataclass
class PlacementSolution:
    """NSGA-II placement solution (chromosome)"""

    # Node assignment vector: job_id -> node_id
    assignments: dict[str, str] = field(default_factory=dict)

    # Objective values (to minimize)
    latency_objective: float = float("inf")
    load_balance_objective: float = float("inf")
    trust_objective: float = float("inf")  # Negative trust (to minimize)
    cost_objective: float = float("inf")
    marketplace_price_objective: float = float("inf")  # Marketplace pricing

    # NSGA-II metrics
    dominance_rank: int = 0
    crowding_distance: float = 0.0

    # Solution metadata
    feasible: bool = True
    constraint_violations: list[str] = field(default_factory=list)

    def dominates(self, other: "PlacementSolution") -> bool:
        """Check if this solution dominates another (Pareto dominance)"""
        objectives_self = [
            self.latency_objective,
            self.load_balance_objective,
            self.trust_objective,
            self.cost_objective,
            self.marketplace_price_objective,
        ]
        objectives_other = [
            other.latency_objective,
            other.load_balance_objective,
            other.trust_objective,
            other.cost_objective,
            other.marketplace_price_objective,
        ]

        # At least one objective must be strictly better
        at_least_one_better = any(s < o for s, o in zip(objectives_self, objectives_other))

        # All objectives must be no worse
        all_no_worse = all(s <= o for s, o in zip(objectives_self, objectives_other))

        return at_least_one_better and all_no_worse

    def copy(self) -> "PlacementSolution":
        """Create a deep copy of the solution"""
        return PlacementSolution(
            assignments=self.assignments.copy(),
            latency_objective=self.latency_objective,
            load_balance_objective=self.load_balance_objective,
            trust_objective=self.trust_objective,
            cost_objective=self.cost_objective,
            marketplace_price_objective=self.marketplace_price_objective,
            dominance_rank=self.dominance_rank,
            crowding_distance=self.crowding_distance,
            feasible=self.feasible,
            constraint_violations=self.constraint_violations.copy(),
        )


class NSGA2PlacementEngine:
    """
    NSGA-II Multi-Objective Optimization Engine for Fog Job Placement

    Implements the Non-dominated Sorting Genetic Algorithm II to find
    Pareto-optimal job placements across fog nodes.
    """

    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        tournament_size: int = 3,
    ):
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size

        # Algorithm state
        self.population: list[PlacementSolution] = []
        self.generation = 0
        self.best_solutions: list[PlacementSolution] = []

        # Performance tracking
        self.optimization_history: list[dict[str, Any]] = []

    def optimize(
        self, jobs: list[JobRequest], nodes: list[FogNode], strategy: PlacementStrategy = PlacementStrategy.NSGA_II
    ) -> PlacementSolution:
        """
        Find optimal job placement using NSGA-II or fallback strategy

        Args:
            jobs: List of job requests to place
            nodes: Available fog nodes
            strategy: Placement strategy to use

        Returns:
            Best placement solution found
        """

        start_time = time.time()

        if strategy != PlacementStrategy.NSGA_II:
            # Use simpler strategy for quick placement
            solution = self._simple_placement(jobs, nodes, strategy)
            self._evaluate_solution(solution, jobs, nodes)
            return solution

        # Filter nodes that can handle jobs
        capable_nodes = self._filter_capable_nodes(jobs, nodes)
        if not capable_nodes:
            logger.error("No capable nodes found for job placement")
            return self._create_empty_solution(jobs)

        logger.info(f"Starting NSGA-II optimization for {len(jobs)} jobs across {len(capable_nodes)} nodes")

        # Initialize population
        self.population = self._initialize_population(jobs, capable_nodes)

        # Evolution loop
        for generation in range(self.max_generations):
            self.generation = generation

            # Evaluate all solutions
            for solution in self.population:
                self._evaluate_solution(solution, jobs, capable_nodes)

            # Non-dominated sorting
            self._non_dominated_sort()

            # Calculate crowding distance
            self._calculate_crowding_distance()

            # Check convergence
            if self._check_convergence():
                logger.info(f"NSGA-II converged at generation {generation}")
                break

            # Generate next generation
            new_population = self._generate_offspring(jobs, capable_nodes)

            # Environmental selection
            combined_population = self.population + new_population
            for solution in combined_population:
                self._evaluate_solution(solution, jobs, capable_nodes)

            self._non_dominated_sort(combined_population)
            self._calculate_crowding_distance(combined_population)

            # Select best solutions for next generation
            self.population = self._environmental_selection(combined_population)

            # Track progress
            self._record_generation_stats(jobs, capable_nodes)

        # Select best solution from final Pareto front
        best_solution = self._select_best_solution()

        optimization_time = time.time() - start_time
        logger.info(f"NSGA-II optimization completed in {optimization_time:.3f}s, generation {self.generation}")

        return best_solution

    def _simple_placement(
        self, jobs: list[JobRequest], nodes: list[FogNode], strategy: PlacementStrategy
    ) -> PlacementSolution:
        """Implement simple placement strategies as fallback"""

        solution = PlacementSolution()

        if strategy == PlacementStrategy.LATENCY_FIRST:
            # Sort nodes by latency for each job
            for job in jobs:
                best_node = min(
                    (node for node in nodes if node.can_handle_job(job.to_requirements_dict())),
                    key=lambda n: n.estimate_latency(job.job_size_mb),
                    default=None,
                )
                if best_node:
                    solution.assignments[job.job_id] = best_node.node_id

        elif strategy == PlacementStrategy.LOAD_BALANCE:
            # Sort nodes by utilization
            for job in jobs:
                best_node = min(
                    (node for node in nodes if node.can_handle_job(job.to_requirements_dict())),
                    key=lambda n: n.cpu_utilization + n.memory_utilization,
                    default=None,
                )
                if best_node:
                    solution.assignments[job.job_id] = best_node.node_id

        elif strategy == PlacementStrategy.TRUST_FIRST:
            # Sort nodes by trust score
            for job in jobs:
                best_node = max(
                    (node for node in nodes if node.can_handle_job(job.to_requirements_dict())),
                    key=lambda n: n.trust_score,
                    default=None,
                )
                if best_node:
                    solution.assignments[job.job_id] = best_node.node_id

        elif strategy == PlacementStrategy.COST_OPTIMIZE:
            # Sort nodes by cost
            for job in jobs:
                best_node = min(
                    (node for node in nodes if node.can_handle_job(job.to_requirements_dict())),
                    key=lambda n: n.estimate_cost(job.to_requirements_dict(), job.estimated_duration_hours),
                    default=None,
                )
                if best_node:
                    solution.assignments[job.job_id] = best_node.node_id

        elif strategy == PlacementStrategy.ROUND_ROBIN:
            # Simple round-robin assignment
            capable_nodes = [
                node for node in nodes if any(node.can_handle_job(job.to_requirements_dict()) for job in jobs)
            ]
            for i, job in enumerate(jobs):
                if capable_nodes:
                    node = capable_nodes[i % len(capable_nodes)]
                    solution.assignments[job.job_id] = node.node_id

        return solution

    def _filter_capable_nodes(self, jobs: list[JobRequest], nodes: list[FogNode]) -> list[FogNode]:
        """Filter nodes that can handle at least one job"""
        capable_nodes = []

        for node in nodes:
            if not node.is_healthy:
                continue

            # Check if node can handle any job
            can_handle_any = any(node.can_handle_job(job.to_requirements_dict()) for job in jobs)

            if can_handle_any:
                capable_nodes.append(node)

        return capable_nodes

    def _initialize_population(self, jobs: list[JobRequest], nodes: list[FogNode]) -> list[PlacementSolution]:
        """Initialize random population of placement solutions"""
        population = []

        for _ in range(self.population_size):
            solution = PlacementSolution()

            for job in jobs:
                # Find nodes that can handle this job
                capable_nodes = [
                    node
                    for node in nodes
                    if node.can_handle_job(job.to_requirements_dict()) and node.node_id not in job.excluded_nodes
                ]

                if capable_nodes:
                    # Random node selection
                    chosen_node = random.choice(capable_nodes)
                    solution.assignments[job.job_id] = chosen_node.node_id
                else:
                    # No capable nodes - mark as infeasible
                    solution.feasible = False
                    solution.constraint_violations.append(f"No capable nodes for job {job.job_id}")

            population.append(solution)

        return population

    def _evaluate_solution(self, solution: PlacementSolution, jobs: list[JobRequest], nodes: list[FogNode]):
        """Evaluate solution objectives"""

        if not solution.feasible:
            # Assign worst possible values
            solution.latency_objective = 1e6
            solution.load_balance_objective = 1e6
            solution.trust_objective = 1e6
            solution.cost_objective = 1e6
            solution.marketplace_price_objective = 1e6
            return

        # Create node lookup
        node_lookup = {node.node_id: node for node in nodes}
        job_lookup = {job.job_id: job for job in jobs}

        # Track node loads for load balancing
        node_loads = {node.node_id: 0.0 for node in nodes}

        total_latency = 0.0
        total_cost = 0.0
        total_trust = 0.0
        total_marketplace_price = 0.0
        valid_assignments = 0

        for job_id, node_id in solution.assignments.items():
            job = job_lookup.get(job_id)
            node = node_lookup.get(node_id)

            if not job or not node:
                solution.feasible = False
                continue

            # Latency objective
            latency = node.estimate_latency(job.job_size_mb)
            total_latency += latency

            # Cost objective
            cost = node.estimate_cost(job.to_requirements_dict(), job.estimated_duration_hours)
            total_cost += cost

            # Trust objective (negative to minimize)
            total_trust += node.trust_score

            # Marketplace pricing objective
            marketplace_price = self._calculate_marketplace_price(job, node)
            total_marketplace_price += marketplace_price

            # Check marketplace price constraints
            if job.max_price is not None and marketplace_price > job.max_price:
                solution.constraint_violations.append(
                    f"Job {job_id} marketplace price ${marketplace_price:.4f} exceeds max_price ${job.max_price:.4f}"
                )
                solution.feasible = False

            # Check trust constraints
            if node.trust_score < job.min_trust_score:
                solution.constraint_violations.append(
                    f"Node {node_id} trust score {node.trust_score:.3f} below required {job.min_trust_score:.3f}"
                )
                solution.feasible = False

            # Load balancing - track resource usage
            node_loads[node_id] += job.cpu_cores / node.cpu_cores

            valid_assignments += 1

        if valid_assignments == 0:
            solution.feasible = False
            return

        # Calculate objectives
        solution.latency_objective = total_latency / valid_assignments
        solution.cost_objective = total_cost
        solution.trust_objective = -total_trust / valid_assignments  # Negative to minimize
        solution.marketplace_price_objective = total_marketplace_price

        # Load balance objective - variance in node utilization
        load_values = list(node_loads.values())
        if load_values:
            load_mean = sum(load_values) / len(load_values)
            load_variance = sum((load - load_mean) ** 2 for load in load_values) / len(load_values)
            solution.load_balance_objective = load_variance
        else:
            solution.load_balance_objective = 0.0

    def _calculate_marketplace_price(self, job: JobRequest, node: FogNode) -> float:
        """Calculate marketplace price for job on node"""

        # Basic pricing model based on job requirements and node characteristics
        base_cpu_rate = 0.10  # Base rate per CPU-hour
        base_memory_rate = 0.01  # Base rate per GB-hour

        # Calculate resource-based cost
        cpu_cost = job.cpu_cores * job.estimated_duration_hours * base_cpu_rate
        memory_cost = job.memory_gb * job.estimated_duration_hours * base_memory_rate
        base_cost = cpu_cost + memory_cost

        # Apply trust premium: higher trust nodes charge more
        trust_multiplier = 1.0 + (node.trust_score * 0.5)  # Up to 50% premium

        # Apply pricing tier multiplier based on job class
        tier_multipliers = {
            JobClass.B_CLASS: 1.0,  # Basic tier
            JobClass.A_CLASS: 1.5,  # Standard tier
            JobClass.S_CLASS: 2.0,  # Premium tier
        }
        tier_multiplier = tier_multipliers.get(job.job_class, 1.0)

        # Apply bid type adjustment
        if job.bid_type == BidType.SPOT:
            # Spot pricing - potentially lower but variable
            spot_discount = 0.8  # 20% discount for spot
            bid_multiplier = spot_discount
        else:
            # On-demand pricing - stable but higher
            bid_multiplier = 1.0

        # Apply utilization-based pricing
        utilization_factor = max(node.cpu_utilization, node.memory_utilization)
        utilization_multiplier = 1.0 + (utilization_factor * 0.3)  # Up to 30% premium for high utilization

        # Calculate final marketplace price
        marketplace_price = base_cost * trust_multiplier * tier_multiplier * bid_multiplier * utilization_multiplier

        return round(marketplace_price, 4)

    def _non_dominated_sort(self, population: list[PlacementSolution] | None = None):
        """Perform non-dominated sorting (NSGA-II core algorithm)"""

        if population is None:
            population = self.population

        # Initialize dominance data structures
        dominated_solutions = {i: set() for i in range(len(population))}
        domination_count = {i: 0 for i in range(len(population))}
        fronts = [set()]

        # Calculate dominance relationships
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                if population[i].dominates(population[j]):
                    dominated_solutions[i].add(j)
                    domination_count[j] += 1
                elif population[j].dominates(population[i]):
                    dominated_solutions[j].add(i)
                    domination_count[i] += 1

            # Solutions with no domination belong to first front
            if domination_count[i] == 0:
                population[i].dominance_rank = 0
                fronts[0].add(i)

        # Build subsequent fronts
        front_index = 0
        while fronts[front_index]:
            next_front = set()

            for i in fronts[front_index]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].dominance_rank = front_index + 1
                        next_front.add(j)

            if next_front:
                fronts.append(next_front)
            front_index += 1

    def _calculate_crowding_distance(self, population: list[PlacementSolution] | None = None):
        """Calculate crowding distance for diversity preservation"""

        if population is None:
            population = self.population

        # Reset crowding distances
        for solution in population:
            solution.crowding_distance = 0.0

        if len(population) <= 2:
            return

        # Calculate for each objective
        objectives = [
            "latency_objective",
            "load_balance_objective",
            "trust_objective",
            "cost_objective",
            "marketplace_price_objective",
        ]

        for objective in objectives:
            # Sort by objective value
            sorted_indices = sorted(range(len(population)), key=lambda i: getattr(population[i], objective))

            # Get objective range
            obj_min = getattr(population[sorted_indices[0]], objective)
            obj_max = getattr(population[sorted_indices[-1]], objective)
            obj_range = obj_max - obj_min

            if obj_range == 0:
                continue

            # Boundary solutions get infinite distance
            population[sorted_indices[0]].crowding_distance = float("inf")
            population[sorted_indices[-1]].crowding_distance = float("inf")

            # Calculate distances for intermediate solutions
            for i in range(1, len(sorted_indices) - 1):
                distance = (
                    getattr(population[sorted_indices[i + 1]], objective)
                    - getattr(population[sorted_indices[i - 1]], objective)
                ) / obj_range
                population[sorted_indices[i]].crowding_distance += distance

    def _generate_offspring(self, jobs: list[JobRequest], nodes: list[FogNode]) -> list[PlacementSolution]:
        """Generate offspring population using crossover and mutation"""

        offspring = []

        while len(offspring) < self.population_size:
            # Tournament selection for parents
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()

            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2, jobs)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])

        # Mutation
        for solution in offspring:
            if random.random() < self.mutation_rate:
                self._mutate(solution, jobs, nodes)

        return offspring[: self.population_size]

    def _tournament_selection(self) -> PlacementSolution:
        """Tournament selection for parent choosing"""

        tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))

        # Select best solution based on dominance rank and crowding distance
        best = min(tournament, key=lambda s: (s.dominance_rank, -s.crowding_distance))
        return best

    def _crossover(
        self, parent1: PlacementSolution, parent2: PlacementSolution, jobs: list[JobRequest]
    ) -> tuple[PlacementSolution, PlacementSolution]:
        """Single-point crossover for job assignments"""

        child1 = parent1.copy()
        child2 = parent2.copy()

        if not jobs:
            return child1, child2

        # Random crossover point
        crossover_point = random.randint(0, len(jobs) - 1)
        job_ids = [job.job_id for job in jobs]

        # Swap assignments after crossover point
        for i in range(crossover_point, len(job_ids)):
            job_id = job_ids[i]

            if job_id in parent1.assignments and job_id in parent2.assignments:
                child1.assignments[job_id] = parent2.assignments[job_id]
                child2.assignments[job_id] = parent1.assignments[job_id]

        return child1, child2

    def _mutate(self, solution: PlacementSolution, jobs: list[JobRequest], nodes: list[FogNode]):
        """Mutation operator - randomly reassign jobs to different nodes"""

        job_lookup = {job.job_id: job for job in jobs}

        for job_id in solution.assignments:
            if random.random() < 0.1:  # 10% mutation rate per assignment
                job = job_lookup.get(job_id)
                if job:
                    # Find alternative capable nodes
                    capable_nodes = [
                        node
                        for node in nodes
                        if node.can_handle_job(job.to_requirements_dict())
                        and node.node_id not in job.excluded_nodes
                        and node.node_id != solution.assignments[job_id]
                    ]

                    if capable_nodes:
                        solution.assignments[job_id] = random.choice(capable_nodes).node_id

    def _environmental_selection(self, combined_population: list[PlacementSolution]) -> list[PlacementSolution]:
        """Select best solutions for next generation"""

        selected = []

        # Group by dominance rank
        fronts = {}
        for solution in combined_population:
            rank = solution.dominance_rank
            if rank not in fronts:
                fronts[rank] = []
            fronts[rank].append(solution)

        # Add fronts in order until population is filled
        for rank in sorted(fronts.keys()):
            front = fronts[rank]

            if len(selected) + len(front) <= self.population_size:
                # Add entire front
                selected.extend(front)
            else:
                # Add partial front based on crowding distance
                remaining_slots = self.population_size - len(selected)
                front_sorted = sorted(front, key=lambda s: -s.crowding_distance)
                selected.extend(front_sorted[:remaining_slots])
                break

        return selected

    def _check_convergence(self) -> bool:
        """Check if algorithm has converged"""

        if self.generation < 10:
            return False

        # Check if best solutions haven't improved in last 20 generations
        if len(self.optimization_history) >= 20:
            recent_best = [gen["best_latency"] for gen in self.optimization_history[-20:]]
            improvement = max(recent_best) - min(recent_best)
            return improvement < 0.01  # 1% improvement threshold

        return False

    def _record_generation_stats(self, jobs: list[JobRequest], nodes: list[FogNode]):
        """Record generation statistics for monitoring"""

        # Find best solution in current population
        pareto_front = [s for s in self.population if s.dominance_rank == 0]
        if pareto_front:
            best_latency = min(s.latency_objective for s in pareto_front)
            best_cost = min(s.cost_objective for s in pareto_front)
            best_marketplace_price = min(s.marketplace_price_objective for s in pareto_front)
            avg_trust = sum(s.trust_objective for s in pareto_front) / len(pareto_front)
        else:
            best_latency = float("inf")
            best_cost = float("inf")
            best_marketplace_price = float("inf")
            avg_trust = 0.0

        stats = {
            "generation": self.generation,
            "best_latency": best_latency,
            "best_cost": best_cost,
            "best_marketplace_price": best_marketplace_price,
            "avg_trust": avg_trust,
            "pareto_front_size": len(pareto_front),
            "feasible_solutions": sum(1 for s in self.population if s.feasible),
        }

        self.optimization_history.append(stats)

    def _select_best_solution(self) -> PlacementSolution:
        """Select best solution from final population"""

        # Get Pareto front (rank 0)
        pareto_front = [s for s in self.population if s.dominance_rank == 0 and s.feasible]

        if not pareto_front:
            # Fallback to best rank if no feasible solutions in Pareto front
            best_rank = min(s.dominance_rank for s in self.population if s.feasible)
            pareto_front = [s for s in self.population if s.dominance_rank == best_rank and s.feasible]

        if not pareto_front:
            # Last resort - any feasible solution
            feasible = [s for s in self.population if s.feasible]
            if feasible:
                return feasible[0]
            else:
                return self.population[0]  # Return something even if infeasible

        # From Pareto front, select solution with best balanced objectives
        # Use weighted sum with equal weights
        best_solution = min(
            pareto_front,
            key=lambda s: (
                s.latency_objective / 1000
                + s.load_balance_objective  # Normalize to ~1
                + abs(s.trust_objective)
                + s.cost_objective / 10
                + s.marketplace_price_objective / 10  # Normalize to ~1
            ),
        )  # Marketplace price weight

        return best_solution

    def _create_empty_solution(self, jobs: list[JobRequest]) -> PlacementSolution:
        """Create empty solution when no placement is possible"""
        solution = PlacementSolution()
        solution.feasible = False
        solution.constraint_violations = ["No capable nodes available"]
        return solution


class FogScheduler:
    """
    Main fog job scheduler integrating NSGA-II placement with existing infrastructure
    """

    def __init__(self):
        self.placement_engine = NSGA2PlacementEngine()
        self.node_registry: dict[str, FogNode] = {}
        self.job_queue: list[JobRequest] = []
        self.placement_cache: dict[str, PlacementSolution] = {}

        # Performance metrics
        self.placement_latencies: list[float] = []
        self.successful_placements = 0
        self.failed_placements = 0

        logger.info("Fog scheduler initialized with NSGA-II placement engine")

    async def register_node(self, node: FogNode) -> None:
        """Register a fog node in the scheduler"""
        self.node_registry[node.node_id] = node
        node.last_heartbeat = datetime.now(UTC)
        logger.info(f"Registered fog node: {node.node_id}")

    async def update_node_status(self, node_id: str, status_update: dict[str, Any]) -> None:
        """Update node status and metrics"""
        if node_id in self.node_registry:
            node = self.node_registry[node_id]

            # Update utilization
            if "cpu_utilization" in status_update:
                node.cpu_utilization = status_update["cpu_utilization"]
            if "memory_utilization" in status_update:
                node.memory_utilization = status_update["memory_utilization"]

            # Update performance metrics
            if "avg_latency_ms" in status_update:
                node.avg_latency_ms = status_update["avg_latency_ms"]
            if "success_rate" in status_update:
                node.success_rate = status_update["success_rate"]

            # Update job counts
            if "active_jobs" in status_update:
                node.active_jobs = status_update["active_jobs"]
            if "queued_jobs" in status_update:
                node.queued_jobs = status_update["queued_jobs"]

            node.last_heartbeat = datetime.now(UTC)

    async def schedule_job(
        self, job: JobRequest, strategy: PlacementStrategy = PlacementStrategy.NSGA_II
    ) -> dict[str, Any]:
        """
        Schedule a single job for placement

        Returns:
            Placement result with node assignment and metrics
        """

        start_time = time.time()

        # Get available nodes
        available_nodes = [
            node
            for node in self.node_registry.values()
            if node.is_healthy
            and node.last_heartbeat
            and (datetime.now(UTC) - node.last_heartbeat) < timedelta(minutes=5)
        ]

        if not available_nodes:
            self.failed_placements += 1
            return {
                "success": False,
                "error": "No available fog nodes",
                "job_id": job.job_id,
                "placement_time_ms": (time.time() - start_time) * 1000,
            }

        # Run placement optimization
        try:
            solution = self.placement_engine.optimize([job], available_nodes, strategy)

            placement_time_ms = (time.time() - start_time) * 1000
            self.placement_latencies.append(placement_time_ms)

            if solution.feasible and job.job_id in solution.assignments:
                self.successful_placements += 1
                assigned_node_id = solution.assignments[job.job_id]

                # Update node job count
                if assigned_node_id in self.node_registry:
                    self.node_registry[assigned_node_id].queued_jobs += 1

                return {
                    "success": True,
                    "node_id": assigned_node_id,
                    "job_id": job.job_id,
                    "placement_time_ms": placement_time_ms,
                    "objectives": {
                        "latency_ms": solution.latency_objective,
                        "cost_estimate": solution.cost_objective,
                        "marketplace_price": solution.marketplace_price_objective,
                        "trust_score": -solution.trust_objective,
                        "load_balance": solution.load_balance_objective,
                    },
                    "strategy": strategy.value,
                }
            else:
                self.failed_placements += 1
                return {
                    "success": False,
                    "error": "No feasible placement found",
                    "violations": solution.constraint_violations,
                    "job_id": job.job_id,
                    "placement_time_ms": placement_time_ms,
                }

        except Exception as e:
            self.failed_placements += 1
            logger.error(f"Placement optimization failed for job {job.job_id}: {e}")
            return {
                "success": False,
                "error": f"Placement optimization failed: {str(e)}",
                "job_id": job.job_id,
                "placement_time_ms": (time.time() - start_time) * 1000,
            }

    async def schedule_batch(
        self, jobs: list[JobRequest], strategy: PlacementStrategy = PlacementStrategy.NSGA_II
    ) -> dict[str, Any]:
        """
        Schedule multiple jobs simultaneously for better optimization

        Returns:
            Batch placement result with assignments and metrics
        """

        start_time = time.time()

        # Get available nodes
        available_nodes = [
            node
            for node in self.node_registry.values()
            if node.is_healthy
            and node.last_heartbeat
            and (datetime.now(UTC) - node.last_heartbeat) < timedelta(minutes=5)
        ]

        if not available_nodes:
            return {
                "success": False,
                "error": "No available fog nodes",
                "job_count": len(jobs),
                "placement_time_ms": (time.time() - start_time) * 1000,
            }

        # Run batch placement optimization
        try:
            solution = self.placement_engine.optimize(jobs, available_nodes, strategy)

            placement_time_ms = (time.time() - start_time) * 1000
            self.placement_latencies.append(placement_time_ms)

            successful_assignments = 0
            assignments = {}

            for job in jobs:
                if solution.feasible and job.job_id in solution.assignments:
                    assigned_node_id = solution.assignments[job.job_id]
                    assignments[job.job_id] = assigned_node_id
                    successful_assignments += 1

                    # Update node job count
                    if assigned_node_id in self.node_registry:
                        self.node_registry[assigned_node_id].queued_jobs += 1

            self.successful_placements += successful_assignments
            self.failed_placements += len(jobs) - successful_assignments

            return {
                "success": successful_assignments > 0,
                "assignments": assignments,
                "successful_placements": successful_assignments,
                "failed_placements": len(jobs) - successful_assignments,
                "placement_time_ms": placement_time_ms,
                "objectives": {
                    "avg_latency_ms": solution.latency_objective,
                    "total_cost_estimate": solution.cost_objective,
                    "total_marketplace_price": solution.marketplace_price_objective,
                    "avg_trust_score": -solution.trust_objective,
                    "load_balance_score": solution.load_balance_objective,
                },
                "strategy": strategy.value,
                "pareto_front_size": (
                    len([s for s in self.placement_engine.population if s.dominance_rank == 0])
                    if strategy == PlacementStrategy.NSGA_II
                    else 1
                ),
            }

        except Exception as e:
            self.failed_placements += len(jobs)
            logger.error(f"Batch placement optimization failed: {e}")
            return {
                "success": False,
                "error": f"Batch placement optimization failed: {str(e)}",
                "job_count": len(jobs),
                "placement_time_ms": (time.time() - start_time) * 1000,
            }

    def get_scheduler_stats(self) -> dict[str, Any]:
        """Get comprehensive scheduler performance statistics"""

        p95_latency = 0.0
        avg_latency = 0.0

        if self.placement_latencies:
            sorted_latencies = sorted(self.placement_latencies)
            p95_index = int(0.95 * len(sorted_latencies))
            p95_latency = sorted_latencies[p95_index] if p95_index < len(sorted_latencies) else sorted_latencies[-1]
            avg_latency = sum(self.placement_latencies) / len(self.placement_latencies)

        total_placements = self.successful_placements + self.failed_placements
        success_rate = self.successful_placements / total_placements if total_placements > 0 else 0.0

        return {
            "total_nodes": len(self.node_registry),
            "healthy_nodes": sum(1 for node in self.node_registry.values() if node.is_healthy),
            "total_placements": total_placements,
            "successful_placements": self.successful_placements,
            "failed_placements": self.failed_placements,
            "success_rate": success_rate,
            "avg_placement_latency_ms": avg_latency,
            "p95_placement_latency_ms": p95_latency,
            "queue_length": len(self.job_queue),
            "optimization_generations": len(self.placement_engine.optimization_history),
        }


# Global scheduler instance
_scheduler: FogScheduler | None = None


async def get_scheduler() -> FogScheduler:
    """Get global scheduler instance"""
    global _scheduler

    if _scheduler is None:
        _scheduler = FogScheduler()

    return _scheduler


async def schedule_fog_job(job: JobRequest, strategy: PlacementStrategy = PlacementStrategy.NSGA_II) -> dict[str, Any]:
    """Convenience function to schedule a fog job"""

    scheduler = await get_scheduler()
    return await scheduler.schedule_job(job, strategy)


async def schedule_fog_jobs_batch(
    jobs: list[JobRequest], strategy: PlacementStrategy = PlacementStrategy.NSGA_II
) -> dict[str, Any]:
    """Convenience function to schedule multiple fog jobs"""

    scheduler = await get_scheduler()
    return await scheduler.schedule_batch(jobs, strategy)
