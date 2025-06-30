import httpx


async def _execute_test_data_deletion(unique_identifier):
    async with httpx.AsyncClient() as client:
        await client.delete(f"http://twin:8001/v1/user/{unique_identifier}")
