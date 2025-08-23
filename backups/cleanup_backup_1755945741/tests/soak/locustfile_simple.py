"""Minimal soak test configuration.

Run with 10 users for 5 minutes by default.
"""

from locust import HttpUser, between, task


class QuickUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def health(self):
        self.client.get("/health")
