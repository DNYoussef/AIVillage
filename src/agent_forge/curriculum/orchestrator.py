"""Curriculum Orchestrator - Master coordinator for the complete curriculum pipeline.

Orchestrates all curriculum components to maintain optimal learning flow:
- Queue management and batch processing
- Resource allocation and priority balancing
- Integration with Agent Forge training loop
- Real-time curriculum adaptation
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .controller import EdgeController
from .edge_finder import EdgeFinder
from .grader import Grader
from .hints import HintGenerator
from .mastery import MasteryTracker
from .openrouter import OpenRouterLLM
from .problem_generator import ProblemGenerator
from .schemas import (
    BatchItem,
    BatchOperation,
    ConductorRequest,
    ConductorResponse,
    EdgeConstraints,
    EdgeWindow,
    MasteryStats,
    Problem,
    ProblemVariant,
    QueueBacklog,
)
from .variant_maker import VariantMaker

logger = logging.getLogger(__name__)


class CurriculumOrchestrator:
    """Master orchestrator coordinating all curriculum components."""

    def __init__(
        self,
        llm_client: OpenRouterLLM,
        model: str = "anthropic/claude-3-5-sonnet-20241022",
        temperature: float = 0.3,
        storage_path: str = "curriculum.db",
        max_batch_size: int = 50,
    ):
        """Initialize CurriculumOrchestrator.

        Args:
            llm_client: OpenRouter client for LLM calls
            model: Model to use for orchestration decisions
            temperature: Temperature for orchestration
            storage_path: Path to curriculum state database
            max_batch_size: Maximum items per batch operation
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.storage_path = storage_path
        self.max_batch_size = max_batch_size

        # Load template
        template_path = Path(__file__).parent / "templates" / "conductor.jinja"
        with open(template_path, encoding="utf-8") as f:
            self.template = f.read()

        # Initialize component modules
        self.edge_finder = EdgeFinder(llm_client)
        self.problem_generator = ProblemGenerator(llm_client)
        self.variant_maker = VariantMaker(llm_client)
        self.grader = Grader(llm_client)
        self.hint_generator = HintGenerator(llm_client)
        self.mastery_tracker = MasteryTracker(llm_client, storage_path=storage_path)
        self.edge_controller = EdgeController(llm_client)

        # Curriculum state
        self.current_edge: EdgeWindow | None = None
        self.problem_queue: list[Problem] = []
        self.variant_queue: list[ProblemVariant] = []
        self.hint_variant_queue: list[ProblemVariant] = []

        logger.info("CurriculumOrchestrator initialized with all components")

    async def initialize_curriculum(
        self,
        domain: str,
        initial_telemetry: list[Any],
        constraints: EdgeConstraints | None = None,
    ) -> dict[str, Any]:
        """Initialize curriculum with edge finding and initial problem generation.

        Args:
            domain: Problem domain (e.g., "coding-python")
            initial_telemetry: Initial telemetry data for edge detection
            constraints: Edge finding constraints

        Returns:
            Dictionary with initialization results
        """
        logger.info(f"Initializing curriculum for domain: {domain}")

        if constraints is None:
            constraints = EdgeConstraints(target_low=0.55, target_high=0.75, problem_budget=1000)

        try:
            # Find initial edge
            logger.info("Finding initial edge-of-chaos band...")
            edge_response = await self.edge_finder.find_edge(
                domain=domain, telemetry=initial_telemetry, constraints=constraints
            )

            self.current_edge = edge_response.edge

            # Generate initial problem batch
            logger.info("Generating initial problem batch...")
            problem_response = await self.problem_generator.generate_problems(
                domain=domain,
                edge=edge_response.edge,
                topic_mix=edge_response.topic_mix,
                n=min(50, constraints.problem_budget // 20),  # Start with 5% of budget
            )

            self.problem_queue.extend(problem_response.problems)

            logger.info(f"Curriculum initialized with {len(self.problem_queue)} problems")

            return {
                "success": True,
                "domain": domain,
                "initial_edge": {
                    "low": self.current_edge.low,
                    "high": self.current_edge.high,
                },
                "topic_mix": [{"topic": t.topic, "weight": t.weight} for t in edge_response.topic_mix],
                "initial_problems": len(self.problem_queue),
                "generation_plan": {
                    "total_planned": edge_response.generation_plan.n_total,
                    "per_topic_min": edge_response.generation_plan.per_topic_min,
                    "variant_rate": edge_response.generation_plan.variant_rate,
                },
            }

        except Exception as e:
            logger.error(f"Curriculum initialization failed: {e}")
            return {"success": False, "error": str(e), "domain": domain}

    def get_current_backlog(self) -> QueueBacklog:
        """Get current queue backlogs."""
        return QueueBacklog(
            fresh=len(self.problem_queue),
            variants=len(self.variant_queue),
            hint_variants=len(self.hint_variant_queue),
        )

    async def get_mastery_stats(self) -> MasteryStats:
        """Get current mastery statistics across all students."""

        # This would typically query the mastery database
        # For demo purposes, we'll simulate some stats
        try:
            # In a real implementation, this would aggregate from mastery_tracker
            return MasteryStats(
                learning=15,  # Students still learning
                understood=8,  # Students who've mastered concepts
                stalled=3,  # Students who need intervention
            )
        except Exception as e:
            logger.error(f"Failed to get mastery stats: {e}")
            return MasteryStats(learning=0, understood=0, stalled=0)

    async def execute_batch_plan(self, batch_plan: list[BatchItem], domain: str) -> dict[str, Any]:
        """Execute a batch plan from the conductor.

        Args:
            batch_plan: List of batch operations to execute
            domain: Problem domain

        Returns:
            Dictionary with execution results
        """
        results = {
            "operations_completed": 0,
            "operations_failed": 0,
            "items_processed": 0,
            "details": [],
        }

        for batch_item in batch_plan:
            try:
                logger.info(f"Executing {batch_item.op.value}: {batch_item.n} items")

                if batch_item.op == BatchOperation.GENERATE:
                    await self._execute_generate_batch(batch_item, domain, results)

                elif batch_item.op == BatchOperation.VARIANT:
                    await self._execute_variant_batch(batch_item, results)

                elif batch_item.op == BatchOperation.HINT_VARIANT:
                    await self._execute_hint_variant_batch(batch_item, results)

                elif batch_item.op == BatchOperation.PROMOTE:
                    await self._execute_promote_batch(batch_item, results)

                elif batch_item.op == BatchOperation.DROP:
                    await self._execute_drop_batch(batch_item, results)

                results["operations_completed"] += 1
                results["items_processed"] += batch_item.n

            except Exception as e:
                logger.error(f"Batch operation {batch_item.op.value} failed: {e}")
                results["operations_failed"] += 1
                results["details"].append(
                    {
                        "operation": batch_item.op.value,
                        "status": "failed",
                        "error": str(e),
                    }
                )

        return results

    async def _execute_generate_batch(self, batch_item: BatchItem, domain: str, results: dict[str, Any]) -> None:
        """Execute problem generation batch."""

        if not self.current_edge:
            raise RuntimeError("No current edge defined for generation")

        # Extract parameters
        params = batch_item.params
        edge_low = params.get("edge_low", self.current_edge.low)
        edge_high = params.get("edge_high", self.current_edge.high)

        # Create simple topic mix for generation
        from .schemas import TopicMix

        topic_mix = [TopicMix(topic="mixed", weight=1.0)]

        edge = EdgeWindow(low=edge_low, high=edge_high)

        response = await self.problem_generator.generate_problems(
            domain=domain, edge=edge, topic_mix=topic_mix, n=batch_item.n
        )

        self.problem_queue.extend(response.problems)

        results["details"].append(
            {
                "operation": "generate",
                "status": "success",
                "items": len(response.problems),
                "edge": f"{edge_low:.2f}-{edge_high:.2f}",
            }
        )

    async def _execute_variant_batch(self, batch_item: BatchItem, results: dict[str, Any]) -> None:
        """Execute variant generation batch."""

        if not self.problem_queue:
            raise RuntimeError("No problems available for variant generation")

        from .schemas import NumericJitterPolicy, VariantPolicy

        # Default variant policy
        variant_policy = VariantPolicy(paraphrase=True, numeric_jitter=NumericJitterPolicy(enabled=True, pct=10))

        # Select problems for variant generation
        problems_to_vary = self.problem_queue[: batch_item.n]

        generated_variants = []
        for problem in problems_to_vary:
            try:
                response = await self.variant_maker.create_variants(
                    base_problem=problem,
                    variant_policy=variant_policy,
                    n_variants=2,  # Generate 2 variants per problem
                )
                generated_variants.extend(response.variants)
            except Exception as e:
                logger.warning(f"Failed to create variants for {problem.id}: {e}")

        self.variant_queue.extend(generated_variants)

        results["details"].append(
            {
                "operation": "variant",
                "status": "success",
                "items": len(generated_variants),
                "base_problems": len(problems_to_vary),
            }
        )

    async def _execute_hint_variant_batch(self, batch_item: BatchItem, results: dict[str, Any]) -> None:
        """Execute hint variant generation batch."""

        # For hint variants, we'd typically identify problems where students are struggling
        # and create hint-supported versions. For now, simulate this process.

        problems_needing_hints = self.problem_queue[: batch_item.n]

        # In a real implementation, this would:
        # 1. Identify students who need hints from mastery tracker
        # 2. Generate appropriate hints for their wrong answers
        # 3. Create variants with embedded hints

        hint_variants = []
        for problem in problems_needing_hints:
            # Simulate creating hint variant (in reality, would use actual wrong answers)
            try:
                from .schemas import ProblemVariant

                hint_variant = ProblemVariant(
                    id=f"{problem.id}_hint",
                    statement=f"HINT: Start by considering the basic case.\n\n{problem.statement}",
                    canonical_answer=problem.canonical_answer,
                    rubric=problem.rubric,
                    unit_tests=problem.unit_tests,
                )
                hint_variants.append(hint_variant)
            except Exception as e:
                logger.warning(f"Failed to create hint variant for {problem.id}: {e}")

        self.hint_variant_queue.extend(hint_variants)

        results["details"].append(
            {
                "operation": "hint_variant",
                "status": "success",
                "items": len(hint_variants),
            }
        )

    async def _execute_promote_batch(self, batch_item: BatchItem, results: dict[str, Any]) -> None:
        """Execute student promotion batch."""

        # In a real implementation, this would:
        # 1. Identify students ready for promotion from mastery tracker
        # 2. Move them to harder problems
        # 3. Update their learning paths

        # For demo, just log the operation
        logger.info(f"Promoting {batch_item.n} students to harder problems")

        results["details"].append({"operation": "promote", "status": "success", "items": batch_item.n})

    async def _execute_drop_batch(self, batch_item: BatchItem, results: dict[str, Any]) -> None:
        """Execute problem drop batch."""

        # Remove ineffective or surplus problems
        items_to_drop = min(batch_item.n, len(self.problem_queue))

        if items_to_drop > 0:
            # Drop from the end (oldest problems)
            dropped = self.problem_queue[-items_to_drop:]
            self.problem_queue = self.problem_queue[:-items_to_drop]

            logger.info(f"Dropped {len(dropped)} problems from queue")

        results["details"].append({"operation": "drop", "status": "success", "items": items_to_drop})

    async def conduct_batch_planning(
        self,
        capacity: int = 50,
        use_llm_conductor: bool = True,
        use_local_fallback: bool = True,
    ) -> ConductorResponse:
        """Plan next batch of operations based on system state.

        Args:
            capacity: Processing capacity for this batch
            use_llm_conductor: Whether to use LLM for batch planning
            use_local_fallback: Use local planning if LLM fails

        Returns:
            ConductorResponse with batch plan
        """
        logger.info(f"Planning batch operations with capacity {capacity}")

        # Get current system state
        backlog = self.get_current_backlog()
        mastery_stats = await self.get_mastery_stats()

        if not self.current_edge:
            # Create default edge if none exists
            self.current_edge = EdgeWindow(low=0.55, high=0.75)

        # Try LLM conductor
        if use_llm_conductor:
            try:
                request = ConductorRequest(
                    edge=self.current_edge,
                    backlog=backlog,
                    mastery_stats=mastery_stats,
                    capacity=capacity,
                )

                # Render prompt
                prompt = self.llm_client.render_template(
                    self.template,
                    edge=request.edge,
                    backlog=request.backlog,
                    mastery_stats=request.mastery_stats,
                    capacity=request.capacity,
                )

                response = await self.llm_client.invoke_with_schema(
                    prompt=prompt,
                    schema_class=ConductorResponse,
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=2048,
                    max_schema_retries=2,
                )

                logger.info(f"LLM conductor planned {len(response.queue)} operations")
                return response

            except Exception as e:
                logger.error(f"LLM conductor failed: {e}")

        # Fallback to local planning
        if use_local_fallback:
            logger.info("Using local batch planning")
            return self._plan_batch_locally(backlog, mastery_stats, capacity)

        # Return empty plan if all fails
        return ConductorResponse(ok=False, msg="no planning available", queue=[])

    def _plan_batch_locally(
        self, backlog: QueueBacklog, mastery_stats: MasteryStats, capacity: int
    ) -> ConductorResponse:
        """Plan batch operations using local rules."""

        batch_items = []
        remaining_capacity = capacity

        # Priority 1: Hint variants for stalled students (30% of capacity)
        if mastery_stats.stalled > 0 and remaining_capacity > 0:
            hint_capacity = min(int(capacity * 0.3), mastery_stats.stalled, remaining_capacity)
            if hint_capacity > 0:
                batch_items.append(
                    BatchItem(
                        op=BatchOperation.HINT_VARIANT,
                        n=hint_capacity,
                        params={"priority": "high", "target": "stalled"},
                    )
                )
                remaining_capacity -= hint_capacity

        # Priority 2: Generate fresh problems if queue is low (40% of capacity)
        if backlog.fresh < 20 and remaining_capacity > 0:
            generate_capacity = min(int(capacity * 0.4), remaining_capacity)
            if generate_capacity > 0:
                batch_items.append(
                    BatchItem(
                        op=BatchOperation.GENERATE,
                        n=generate_capacity,
                        params={
                            "edge_low": self.current_edge.low,
                            "edge_high": self.current_edge.high,
                        },
                    )
                )
                remaining_capacity -= generate_capacity

        # Priority 3: Create variants for practice (20% of capacity)
        if backlog.variants < 10 and remaining_capacity > 0:
            variant_capacity = min(int(capacity * 0.2), remaining_capacity)
            if variant_capacity > 0:
                batch_items.append(
                    BatchItem(
                        op=BatchOperation.VARIANT,
                        n=variant_capacity,
                        params={"target": "learning"},
                    )
                )
                remaining_capacity -= variant_capacity

        # Priority 4: Promote successful students (10% of capacity)
        if mastery_stats.understood > 0 and remaining_capacity > 0:
            promote_capacity = min(int(capacity * 0.1), mastery_stats.understood, remaining_capacity)
            if promote_capacity > 0:
                batch_items.append(
                    BatchItem(
                        op=BatchOperation.PROMOTE,
                        n=promote_capacity,
                        params={"target": "understood"},
                    )
                )
                remaining_capacity -= promote_capacity

        return ConductorResponse(
            ok=True,
            msg=f"local batch plan ({len(batch_items)} operations)",
            queue=batch_items,
        )

    async def run_curriculum_cycle(self, domain: str, num_cycles: int = 5, cycle_capacity: int = 50) -> dict[str, Any]:
        """Run multiple curriculum cycles for continuous adaptation.

        Args:
            domain: Problem domain
            num_cycles: Number of cycles to run
            cycle_capacity: Processing capacity per cycle

        Returns:
            Dictionary with cycle execution results
        """
        logger.info(f"Running {num_cycles} curriculum cycles")

        cycle_results = []

        for cycle in range(num_cycles):
            logger.info(f"Starting curriculum cycle {cycle + 1}/{num_cycles}")

            try:
                # Plan batch operations
                conductor_response = await self.conduct_batch_planning(capacity=cycle_capacity)

                if not conductor_response.ok or not conductor_response.queue:
                    logger.warning(f"Cycle {cycle + 1}: No operations planned")
                    continue

                # Execute batch plan
                execution_results = await self.execute_batch_plan(conductor_response.queue, domain)

                # Record cycle results
                cycle_result = {
                    "cycle": cycle + 1,
                    "planned_operations": len(conductor_response.queue),
                    "completed_operations": execution_results["operations_completed"],
                    "failed_operations": execution_results["operations_failed"],
                    "items_processed": execution_results["items_processed"],
                    "queue_state": {
                        "fresh_problems": len(self.problem_queue),
                        "variants": len(self.variant_queue),
                        "hint_variants": len(self.hint_variant_queue),
                    },
                }

                cycle_results.append(cycle_result)
                logger.info(f"Cycle {cycle + 1} completed: {execution_results['operations_completed']} operations")

                # Brief pause between cycles
                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Cycle {cycle + 1} failed: {e}")
                cycle_results.append({"cycle": cycle + 1, "error": str(e), "status": "failed"})

        # Calculate summary statistics
        total_operations = sum(r.get("completed_operations", 0) for r in cycle_results)
        total_items = sum(r.get("items_processed", 0) for r in cycle_results)
        failed_cycles = sum(1 for r in cycle_results if "error" in r)

        return {
            "total_cycles": num_cycles,
            "successful_cycles": num_cycles - failed_cycles,
            "failed_cycles": failed_cycles,
            "total_operations_completed": total_operations,
            "total_items_processed": total_items,
            "final_queue_state": {
                "fresh_problems": len(self.problem_queue),
                "variants": len(self.variant_queue),
                "hint_variants": len(self.hint_variant_queue),
            },
            "cycle_details": cycle_results,
        }

    async def get_curriculum_status(self) -> dict[str, Any]:
        """Get comprehensive curriculum status."""

        backlog = self.get_current_backlog()
        mastery_stats = await self.get_mastery_stats()

        return {
            "current_edge": {
                "low": self.current_edge.low if self.current_edge else None,
                "high": self.current_edge.high if self.current_edge else None,
                "width": (self.current_edge.high - self.current_edge.low) if self.current_edge else None,
            },
            "queues": {
                "fresh_problems": backlog.fresh,
                "variants": backlog.variants,
                "hint_variants": backlog.hint_variants,
                "total_queued": backlog.fresh + backlog.variants + backlog.hint_variants,
            },
            "student_distribution": {
                "learning": mastery_stats.learning,
                "understood": mastery_stats.understood,
                "stalled": mastery_stats.stalled,
                "total": mastery_stats.learning + mastery_stats.understood + mastery_stats.stalled,
            },
            "system_health": self._assess_system_health(backlog, mastery_stats),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _assess_system_health(self, backlog: QueueBacklog, mastery_stats: MasteryStats) -> str:
        """Assess overall curriculum system health."""

        total_students = mastery_stats.learning + mastery_stats.understood + mastery_stats.stalled
        total_queued = backlog.fresh + backlog.variants + backlog.hint_variants

        # Check for critical issues
        if total_queued == 0:
            return "critical - no problems queued"

        if total_students > 0:
            stall_rate = mastery_stats.stalled / total_students
            if stall_rate > 0.5:
                return "poor - high stall rate"
            elif stall_rate > 0.3:
                return "fair - moderate stall rate"

        # Check queue balance
        if backlog.fresh < 10:
            return "warning - low fresh problem supply"

        if mastery_stats.stalled > 0 and backlog.hint_variants == 0:
            return "warning - stalled students need hint variants"

        # System looks healthy
        return "good"


async def run_full_curriculum_pipeline(
    api_key: str,
    domain: str,
    initial_telemetry: list[Any],
    num_cycles: int = 3,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function to run complete curriculum pipeline.

    Args:
        api_key: OpenRouter API key
        domain: Problem domain
        initial_telemetry: Initial telemetry for edge detection
        num_cycles: Number of curriculum cycles to run
        **kwargs: Additional arguments for CurriculumOrchestrator

    Returns:
        Dictionary with complete pipeline results
    """
    async with OpenRouterLLM(api_key=api_key) as client:
        orchestrator = CurriculumOrchestrator(client, **kwargs)

        # Initialize curriculum
        init_result = await orchestrator.initialize_curriculum(domain, initial_telemetry)

        if not init_result["success"]:
            return init_result

        # Run curriculum cycles
        cycle_result = await orchestrator.run_curriculum_cycle(domain, num_cycles)

        # Get final status
        final_status = await orchestrator.get_curriculum_status()

        return {
            "initialization": init_result,
            "cycles": cycle_result,
            "final_status": final_status,
            "pipeline_success": True,
        }


if __name__ == "__main__":
    # Demo usage
    import asyncio
    import os
    import random

    async def demo():
        # Generate mock telemetry data
        from .schemas import TelemetryEntry

        telemetry = []
        for i in range(30):
            difficulty = random.uniform(0.3, 0.8)
            # Simulate edge-of-chaos at 0.55-0.75
            if 0.55 <= difficulty <= 0.75:
                correct_prob = 0.65  # Target range
            elif difficulty < 0.45 or difficulty > 0.8:
                correct_prob = 0.30  # Too easy/hard
            else:
                correct_prob = 0.50  # Transition zones

            correct = random.random() < correct_prob
            telemetry.append(TelemetryEntry(task_id=f"demo_task_{i:03d}", difficulty=difficulty, correct=correct))

        api_key = os.getenv("OPENROUTER_API_KEY", "demo-key")

        if api_key == "demo-key":
            print("🔧 Demo mode: Testing curriculum orchestration locally")

            dummy_client = OpenRouterLLM(api_key="dummy")
            orchestrator = CurriculumOrchestrator(dummy_client)

            # Test local batch planning
            from .schemas import MasteryStats, QueueBacklog

            backlog = QueueBacklog(fresh=5, variants=2, hint_variants=0)
            mastery_stats = MasteryStats(learning=10, understood=3, stalled=2)

            print("📊 System State:")
            print(f"   Fresh problems: {backlog.fresh}")
            print(f"   Variants: {backlog.variants}")
            print(f"   Hint variants: {backlog.hint_variants}")
            print(
                f"   Students - Learning: {mastery_stats.learning}, Understood: {mastery_stats.understood}, Stalled: {mastery_stats.stalled}"
            )

            # Plan batch
            response = orchestrator._plan_batch_locally(backlog, mastery_stats, capacity=20)

            print(f"\n🎯 Local Batch Plan ({len(response.queue)} operations):")
            for item in response.queue:
                print(f"   • {item.op.value}: {item.n} items")

            return

        # Live API test
        print("🚀 Testing full curriculum pipeline...")

        try:
            result = await run_full_curriculum_pipeline(
                api_key=api_key,
                domain="coding-python",
                initial_telemetry=telemetry,
                num_cycles=2,  # Keep cycles low for demo
            )

            print("✅ Pipeline execution complete")
            print(f"   Initialization: {'✅' if result['initialization']['success'] else '❌'}")
            print(f"   Cycles run: {result['cycles']['successful_cycles']}/{result['cycles']['total_cycles']}")
            print(f"   Operations completed: {result['cycles']['total_operations_completed']}")
            print(f"   Items processed: {result['cycles']['total_items_processed']}")
            print(f"   Final queues: {result['final_status']['queues']}")
            print(f"   System health: {result['final_status']['system_health']}")

        except Exception as e:
            print(f"❌ Demo failed: {e}")

    asyncio.run(demo())
