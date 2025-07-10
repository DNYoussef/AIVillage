from locust import HttpUser, task, between

class QuickUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def health(self):
        self.client.get("/health")
