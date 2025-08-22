"""
Example: Using the AIVillage Fog Computing SDK

This example demonstrates how to use the Python SDK to interact
with the fog computing platform for job submission, sandbox management,
and usage tracking.
"""

import asyncio
import logging

from packages.fog.sdk import FogClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def job_submission_example():
    """Example: Submit and monitor fog jobs"""

    async with FogClient(
        base_url="http://localhost:8000", api_key="your-api-key-here", namespace="myorg/dev"
    ) as client:
        print("🚀 Job Submission Example")
        print("=" * 40)

        # Simple job submission
        print("1. Submitting a simple Python job...")
        job = await client.submit_job(
            image="python:3.11-alpine",
            args=["python", "-c", "print('Hello from fog!')"],
            resources={"cpu_cores": 1.0, "memory_mb": 512, "max_duration_s": 60},
            labels={"example": "sdk-demo", "type": "hello-world"},
        )
        print(f"   ✓ Job submitted: {job.job_id}")

        # Wait for completion
        print("2. Waiting for job completion...")
        result = await client.wait_for_job(job.job_id, timeout=120.0)
        print(f"   ✓ Job completed with status: {result.status}")
        print(f"   ✓ Exit code: {result.exit_code}")

        # Get logs
        logs = await client.get_job_logs(job.job_id)
        print(f"   ✓ Logs: {logs}")

        # Submit a more complex job
        print("3. Submitting data processing job...")
        data_job = await client.submit_job(
            image="alpine:latest",
            args=["sh", "-c", "echo 'Processing data...' && sleep 5 && echo 'Done'"],
            env={"WORKER_ID": "fog-001", "BATCH_SIZE": "100"},
            resources={"cpu_cores": 2.0, "memory_mb": 1024, "max_duration_s": 300},
            priority="A",  # Standard priority with replication
        )
        print(f"   ✓ Data job submitted: {data_job.job_id}")

        # Monitor progress
        for i in range(10):
            job_status = await client.get_job(data_job.job_id)
            print(f"   📊 Job {data_job.job_id}: {job_status.status}")

            if job_status.status in ["completed", "failed", "cancelled"]:
                break

            await asyncio.sleep(2)

        # List recent jobs
        print("4. Listing recent jobs...")
        recent_jobs = await client.list_jobs(limit=5)
        for job in recent_jobs:
            print(f"   📝 {job.job_id}: {job.status}")


async def sandbox_management_example():
    """Example: Interactive sandbox management"""

    async with FogClient(
        base_url="http://localhost:8000", api_key="your-api-key-here", namespace="myorg/dev"
    ) as client:
        print("\n🏗️ Sandbox Management Example")
        print("=" * 40)

        # Create interactive sandbox
        print("1. Creating Python development sandbox...")
        sandbox = await client.create_sandbox(
            image="python:3.11-slim",
            sandbox_type="interactive",
            resources={"cpu_cores": 1.0, "memory_mb": 2048, "disk_mb": 10240},
            env={"PYTHONPATH": "/workspace", "EDITOR": "nano"},
            network_access=True,  # Allow network for pip installs
        )
        print(f"   ✓ Sandbox created: {sandbox.sandbox_id}")
        print(f"   ✓ Connection URL: {sandbox.connection_url}")
        print(f"   ✓ SSH command: {sandbox.ssh_command}")

        # Execute commands in sandbox
        print("2. Executing commands in sandbox...")

        # Install a package
        result1 = await client.exec_in_sandbox(sandbox.sandbox_id, "pip", args=["install", "numpy"], timeout=60)
        print(f"   ✓ pip install: exit_code={result1.get('exit_code', 'N/A')}")

        # Run Python code
        result2 = await client.exec_in_sandbox(
            sandbox.sandbox_id,
            "python",
            args=["-c", "import numpy as np; print(f'NumPy version: {np.__version__}')"],
            timeout=30,
        )
        print(f"   ✓ Python script: {result2.get('stdout', 'No output')}")

        # Check sandbox status
        print("3. Checking sandbox status...")
        status = await client.get_sandbox(sandbox.sandbox_id)
        print(f"   📊 Status: {status.status}")
        print(f"   ⏰ Idle time: {status.idle_minutes} minutes")

        # List all sandboxes
        print("4. Listing active sandboxes...")
        sandboxes = await client.list_sandboxes(status="active")
        for sb in sandboxes:
            print(f"   📦 {sb.sandbox_id}: {sb.status}")

        # Clean up
        print("5. Cleaning up sandbox...")
        await client.delete_sandbox(sandbox.sandbox_id)
        print("   ✓ Sandbox deleted")


async def usage_tracking_example():
    """Example: Usage and billing tracking"""

    async with FogClient(
        base_url="http://localhost:8000", api_key="your-api-key-here", namespace="myorg/dev"
    ) as client:
        print("\n💰 Usage Tracking Example")
        print("=" * 40)

        # Get current usage
        print("1. Getting current usage metrics...")
        usage = await client.get_usage(period="day")

        for ns_usage in usage:
            print(f"   📊 Namespace: {ns_usage.namespace}")
            print(f"   💵 Total cost: ${ns_usage.total_cost:.2f}")
            print(f"   ⚡ CPU seconds: {ns_usage.cpu_seconds}")
            print(f"   🧠 Memory MB-hours: {ns_usage.memory_mb_hours}")
            print(f"   🏃 Job executions: {ns_usage.job_executions}")
            print()

        # Get pricing info
        print("2. Getting pricing information...")
        pricing = await client.get_pricing()
        print(f"   💲 CPU per second: ${pricing['cpu_second_price']}")
        print(f"   💲 Memory per MB-hour: ${pricing['memory_mb_hour_price']}")
        print(f"   💲 Job execution: ${pricing['job_execution_price']}")
        print(f"   🎯 Priority multipliers: {pricing['priority_multipliers']}")

        # Check quotas
        print("3. Checking namespace quotas...")
        quotas = await client.get_quotas()

        for quota in quotas:
            print(f"   📈 Namespace: {quota['namespace']}")
            print(f"   🚀 Max concurrent jobs: {quota['max_concurrent_jobs']}")
            print(f"   💾 Max memory MB: {quota['max_memory_mb']}")
            print(f"   💸 Daily cost limit: ${quota['daily_cost_limit']}")
            print(f"   📊 Current usage: ${quota['daily_cost_used']:.2f}")
            print()


async def end_to_end_workflow():
    """Example: Complete fog computing workflow"""

    async with FogClient(
        base_url="http://localhost:8000", api_key="your-api-key-here", namespace="myorg/production"
    ) as client:
        print("\n🔄 End-to-End Workflow Example")
        print("=" * 40)

        # 1. Create development sandbox
        print("1. Setting up development environment...")
        dev_sandbox = await client.create_sandbox(
            image="ubuntu:22.04", resources={"cpu_cores": 2.0, "memory_mb": 4096}, network_access=True
        )

        # 2. Develop and test code in sandbox
        print("2. Developing application in sandbox...")
        await client.exec_in_sandbox(dev_sandbox.sandbox_id, "apt-get", args=["update", "-y"])

        await client.exec_in_sandbox(
            dev_sandbox.sandbox_id, "apt-get", args=["install", "-y", "python3", "python3-pip"]
        )

        # Create a simple app
        app_code = """
import time
import sys

def process_data(n):
    print(f'Processing {n} items...')
    for i in range(n):
        # Simulate work
        time.sleep(0.1)
        if i % 10 == 0:
            print(f'Progress: {i}/{n}')
    print('Processing complete!')
    return n

if __name__ == '__main__':
    items = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    result = process_data(items)
    print(f'Processed {result} items successfully')
"""

        await client.exec_in_sandbox(
            dev_sandbox.sandbox_id,
            "python3",
            args=["-c", f"with open('/tmp/app.py', 'w') as f: f.write('''{app_code}''')"],
        )

        print("   ✓ Application developed")

        # 3. Test application in sandbox
        print("3. Testing application...")
        test_result = await client.exec_in_sandbox(
            dev_sandbox.sandbox_id, "python3", args=["/tmp/app.py", "20"], timeout=60
        )
        print(f"   ✓ Test completed: {test_result.get('exit_code', 'N/A')}")

        # 4. Deploy to production via job submission
        print("4. Deploying to production...")
        production_job = await client.submit_job(
            image="python:3.11-alpine",
            args=["python", "-c", app_code.replace("\n", "; "), "100"],
            resources={"cpu_cores": 4.0, "memory_mb": 2048, "max_duration_s": 600},
            priority="S",  # Premium priority for production
            labels={"environment": "production", "app": "data-processor", "version": "1.0"},
        )

        print(f"   ✓ Production job deployed: {production_job.job_id}")

        # 5. Monitor production job
        print("5. Monitoring production deployment...")
        prod_result = await client.wait_for_job(production_job.job_id, timeout=600)
        print(f"   ✓ Production job: {prod_result.status}")

        if prod_result.status == "completed":
            logs = await client.get_job_logs(production_job.job_id)
            print(f"   📝 Production logs: {logs[-200:]}")  # Last 200 chars

        # 6. Clean up development environment
        print("6. Cleaning up development environment...")
        await client.delete_sandbox(dev_sandbox.sandbox_id)
        print("   ✓ Development environment cleaned up")

        # 7. Check final usage
        print("7. Checking resource usage...")
        final_usage = await client.get_usage(namespace="myorg/production")
        if final_usage:
            print(f"   💵 Session cost: ${final_usage[0].total_cost:.4f}")
            print(f"   ⚡ CPU usage: {final_usage[0].cpu_seconds:.1f}s")


async def main():
    """Run all examples"""
    try:
        await job_submission_example()
        await sandbox_management_example()
        await usage_tracking_example()
        await end_to_end_workflow()

        print("\n🎉 All examples completed successfully!")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\n❌ Example failed: {e}")
        print("   Make sure the fog gateway is running on localhost:8000")
        print("   and you have valid API credentials configured.")


if __name__ == "__main__":
    asyncio.run(main())
