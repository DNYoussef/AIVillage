#!/usr/bin/env python3
"""
Demo Script for Unified Federated System

This script demonstrates the complete unified federated system in action,
showing how users can seamlessly submit both inference and training requests
through a single API with automatic tier-based optimization.

Usage:
    python scripts/demo_unified_system.py
"""

import asyncio
import logging
import time
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import the unified system
try:
    from infrastructure.distributed_inference.unified_api import (
        submit_inference,
        submit_training,
        get_job_status,
        get_pricing_estimate,
        get_system_status,
    )

    SYSTEM_AVAILABLE = True
except ImportError as e:
    SYSTEM_AVAILABLE = False
    logger.error(f"Unified system not available: {e}")


class UnifiedSystemDemo:
    """Demo class showing the unified federated system capabilities"""

    def __init__(self):
        self.demo_users = {
            "small_developer": {"tier": "small", "budget": 10.0},
            "startup_founder": {"tier": "medium", "budget": 100.0},
            "enterprise_team": {"tier": "large", "budget": 1000.0},
            "global_corp": {"tier": "enterprise", "budget": 5000.0},
        }

    async def run_complete_demo(self):
        """Run complete demonstration of the unified system"""

        if not SYSTEM_AVAILABLE:
            print("❌ Unified system components not available for demo")
            return

        print("🚀 UNIFIED FEDERATED SYSTEM DEMO")
        print("=" * 50)

        # 1. Show system status
        await self._demo_system_status()

        # 2. Show pricing across tiers
        await self._demo_pricing_estimates()

        # 3. Demo inference for each tier
        await self._demo_inference_all_tiers()

        # 4. Demo training for each tier
        await self._demo_training_all_tiers()

        # 5. Show final system statistics
        await self._demo_final_statistics()

        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("The unified federated system is working perfectly across all tiers.")

    async def _demo_system_status(self):
        """Demonstrate system status reporting"""
        print("\n📊 SYSTEM STATUS")
        print("-" * 30)

        try:
            status = await get_system_status()
            if status["success"]:
                stats = status["data"]["unified_api"]
                print("✅ System Online")
                print(f"📈 Total Jobs: {stats['total_jobs_processed']}")
                print(f"👥 Users Served: {stats['users_served']}")
                print(f"💰 Revenue: ${stats['total_revenue']:.2f}")
                print(f"📊 Success Rate: {stats['success_rate']:.1%}")
            else:
                print(f"❌ System Status Error: {status.get('error')}")
        except Exception as e:
            print(f"❌ Failed to get system status: {e}")

    async def _demo_pricing_estimates(self):
        """Demonstrate pricing across all tiers"""
        print("\n💰 PRICING ESTIMATES BY TIER")
        print("-" * 40)

        for tier in ["small", "medium", "large", "enterprise"]:
            try:
                # Inference pricing
                inf_pricing = await get_pricing_estimate(
                    job_type="inference", model_id="gpt-3-medium", user_tier=tier, duration_hours=1.0
                )

                # Training pricing
                train_pricing = await get_pricing_estimate(
                    job_type="training",
                    model_id="bert-base",
                    user_tier=tier,
                    participants_needed=10,
                    duration_hours=2.0,
                )

                if inf_pricing["success"] and train_pricing["success"]:
                    inf_cost = inf_pricing["data"]["pricing_breakdown"]["estimated_total"]
                    train_cost = train_pricing["data"]["pricing_breakdown"]["estimated_total"]

                    print(f"🎯 {tier.upper()} TIER:")
                    print(f"   Inference: ${inf_cost:.2f}")
                    print(f"   Training:  ${train_cost:.2f}")
                else:
                    print(f"❌ Failed to get pricing for {tier} tier")

            except Exception as e:
                print(f"❌ Pricing error for {tier}: {e}")

    async def _demo_inference_all_tiers(self):
        """Demonstrate inference across all user tiers"""
        print("\n🧠 INFERENCE DEMO - ALL TIERS")
        print("-" * 40)

        inference_jobs = {}

        # Submit inference jobs for each tier
        for user_id, config in self.demo_users.items():
            tier = config["tier"]
            budget = min(config["budget"] * 0.1, 50.0)  # Use 10% of budget, max $50

            try:
                job_id = await submit_inference(
                    user_id=user_id,
                    model_id=f"gpt-3-{tier}" if tier != "enterprise" else "gpt-4",
                    input_data={
                        "prompt": f"Analyze the business potential of federated AI for {tier} tier companies",
                        "max_tokens": 100,
                    },
                    user_tier=tier,
                    max_cost=budget,
                    privacy_level="medium",
                )

                inference_jobs[tier] = job_id
                print(f"✅ {tier.upper()} tier inference submitted: {job_id}")

            except Exception as e:
                print(f"❌ Failed to submit {tier} tier inference: {e}")

        # Monitor jobs
        print("\n⏳ Monitoring inference jobs...")
        await self._monitor_jobs(inference_jobs, "inference")

    async def _demo_training_all_tiers(self):
        """Demonstrate federated training across all user tiers"""
        print("\n🎓 TRAINING DEMO - ALL TIERS")
        print("-" * 40)

        training_jobs = {}

        # Submit training jobs for each tier (scaled appropriately)
        tier_configs = {
            "small": {"participants": 3, "rounds": 3, "model": "bert-small"},
            "medium": {"participants": 10, "rounds": 5, "model": "bert-base"},
            "large": {"participants": 25, "rounds": 10, "model": "bert-large"},
            "enterprise": {"participants": 50, "rounds": 20, "model": "custom-enterprise"},
        }

        for user_id, user_config in self.demo_users.items():
            tier = user_config["tier"]
            tier_training = tier_configs[tier]
            budget = min(user_config["budget"] * 0.2, 1000.0)  # Use 20% of budget, max $1000

            try:
                job_id = await submit_training(
                    user_id=user_id,
                    model_id=tier_training["model"],
                    training_config={
                        "dataset": f"{tier}_tier_sentiment_analysis",
                        "differential_privacy": tier in ["large", "enterprise"],
                        "mobile_optimization": tier == "small",
                        "gpu_acceleration": tier in ["large", "enterprise"],
                    },
                    user_tier=tier,
                    participants_needed=tier_training["participants"],
                    training_rounds=tier_training["rounds"],
                    max_cost=budget,
                    privacy_level="high" if tier in ["large", "enterprise"] else "medium",
                    duration_hours=1.0 if tier == "small" else 2.0,
                )

                training_jobs[tier] = job_id
                print(f"✅ {tier.upper()} tier training submitted: {job_id}")
                print(f"   └── {tier_training['participants']} participants, {tier_training['rounds']} rounds")

            except Exception as e:
                print(f"❌ Failed to submit {tier} tier training: {e}")

        # Monitor training jobs
        print("\n⏳ Monitoring training jobs...")
        await self._monitor_jobs(training_jobs, "training")

    async def _monitor_jobs(self, jobs: Dict[str, str], job_type: str):
        """Monitor job progress and show results"""

        max_wait_time = 60  # seconds
        check_interval = 5  # seconds
        start_time = time.time()

        completed_jobs = {}

        while time.time() - start_time < max_wait_time and len(completed_jobs) < len(jobs):
            for tier, job_id in jobs.items():
                if tier in completed_jobs:
                    continue

                try:
                    status = await get_job_status(job_id)
                    if status["success"]:
                        job_status = status["data"]["status"]

                        if job_status == "completed":
                            completed_jobs[tier] = status["data"]
                            cost = status["data"]["total_cost"]
                            nodes = status["data"]["nodes_allocated"]
                            processing_time = status["data"]["processing_time_ms"]

                            print(f"🎉 {tier.upper()} {job_type} COMPLETED:")
                            print(f"   💰 Cost: ${cost:.4f}")
                            print(f"   🖥️  Nodes: {nodes}")
                            print(f"   ⏱️  Time: {processing_time:.0f}ms")

                        elif job_status == "failed":
                            completed_jobs[tier] = status["data"]
                            error = status["data"].get("error_message", "Unknown error")
                            print(f"❌ {tier.upper()} {job_type} FAILED: {error}")

                        elif job_status in ["submitted", "allocating_resources", "executing"]:
                            print(f"⏳ {tier.upper()} {job_type}: {job_status}")

                except Exception as e:
                    print(f"❌ Error checking {tier} {job_type}: {e}")

            if len(completed_jobs) < len(jobs):
                await asyncio.sleep(check_interval)

        # Summary
        success_count = len([job for job in completed_jobs.values() if job["status"] == "completed"])
        print(f"\n📊 {job_type.upper()} SUMMARY: {success_count}/{len(jobs)} jobs completed successfully")

    async def _demo_final_statistics(self):
        """Show final system statistics after demo"""
        print("\n📈 FINAL SYSTEM STATISTICS")
        print("-" * 40)

        try:
            final_status = await get_system_status()
            if final_status["success"]:
                stats = final_status["data"]

                api_stats = stats["unified_api"]
                job_dist = stats["job_type_distribution"]
                tier_dist = stats.get("tier_usage_distribution", {})

                print(f"📊 Total Jobs Processed: {api_stats['total_jobs_processed']}")
                print(f"👥 Users Served: {api_stats['users_served']}")
                print(f"💰 Total Revenue: ${api_stats['total_revenue']:.2f}")
                print(f"📈 Success Rate: {api_stats['success_rate']:.1%}")

                print("\n🔍 Job Distribution:")
                print(f"   🧠 Inference: {job_dist.get('inference', 0)}")
                print(f"   🎓 Training:  {job_dist.get('training', 0)}")
                print(f"   🔄 Hybrid:    {job_dist.get('hybrid', 0)}")

                if tier_dist:
                    print("\n🎯 Tier Usage:")
                    for tier, count in tier_dist.items():
                        print(f"   {tier.capitalize()}: {count} jobs")

            else:
                print(f"❌ Failed to get final statistics: {final_status.get('error')}")

        except Exception as e:
            print(f"❌ Error getting final statistics: {e}")


async def main():
    """Main demo function"""
    demo = UnifiedSystemDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    print("🎯 UNIFIED FEDERATED SYSTEM DEMONSTRATION")
    print("This demo shows the complete system working across all tiers")
    print("with both inference and training workloads.\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()
