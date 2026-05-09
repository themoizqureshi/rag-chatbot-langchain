"""
Stage 6c — Load test with Locust.

Simulates concurrent users hitting /chat.
Target SLA: p95 latency < 5 seconds under 50 concurrent users.

Pre-requisites:
  1. API server running: uvicorn src.api:app --reload --port 8001
  2. A valid session_id from /ingest (set SESSION_ID env var or update below)

Run:
    pip install locust
    SESSION_ID=<your-session-id> locust -f locust/locustfile.py \
        --host http://localhost:8001 --users 50 --spawn-rate 5 --run-time 60s --headless

Or open the Locust web UI (port 8089) and run from the browser:
    SESSION_ID=<id> locust -f locust/locustfile.py --host http://localhost:8001
"""

import os
import random

from locust import HttpUser, between, task, events

# Set SESSION_ID env var or replace this default with a real session_id from /ingest
SESSION_ID = os.getenv("SESSION_ID", "REPLACE_WITH_SESSION_ID")

SAMPLE_QUESTIONS = [
    "What is the main topic of this document?",
    "Summarize the key findings.",
    "What methods are described?",
    "What are the limitations mentioned?",
    "How does this relate to real-world applications?",
    "What conclusions does the author draw?",
    "What datasets or examples are referenced?",
    "Explain the architecture described in the document.",
    "What are the future directions suggested?",
    "What problem does this document address?",
]


class RAGUser(HttpUser):
    """Simulates a user asking questions against a pre-ingested document."""

    # Wait 1–3 seconds between requests (realistic user think time)
    wait_time = between(1, 3)

    @task(10)
    def ask_question(self):
        question = random.choice(SAMPLE_QUESTIONS)
        with self.client.post(
            "/chat",
            json={"question": question, "session_id": SESSION_ID},
            catch_response=True,
            name="/chat",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if not data.get("answer"):
                    resp.failure("Response missing 'answer' field")
                elif len(data["answer"]) < 10:
                    resp.failure("Answer suspiciously short")
                else:
                    resp.success()
            elif resp.status_code == 404:
                resp.failure(f"Session not found — update SESSION_ID env var")
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    if SESSION_ID == "REPLACE_WITH_SESSION_ID":
        print("\n⚠️  WARNING: SESSION_ID is not set. Set SESSION_ID env var to a valid ingest session ID.")
        print("   Run: curl -X POST http://localhost:8001/ingest -F 'file=@your.pdf'")
        print("   Then: export SESSION_ID=<returned session_id>\n")


@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    stats = environment.stats
    p95 = stats.total.get_response_time_percentile(0.95)
    failure_rate = stats.total.fail_ratio * 100

    print("\n" + "=" * 50)
    print("LOAD TEST SUMMARY")
    print("=" * 50)
    print(f"  Requests:     {stats.total.num_requests:,}")
    print(f"  Failures:     {stats.total.num_failures:,} ({failure_rate:.1f}%)")
    print(f"  p50 latency:  {stats.total.get_response_time_percentile(0.50):.0f}ms")
    print(f"  p95 latency:  {p95:.0f}ms")
    print(f"  p99 latency:  {stats.total.get_response_time_percentile(0.99):.0f}ms")
    print(f"  Avg RPS:      {stats.total.total_rps:.1f}")

    sla_target_ms = 5000
    if p95 and p95 > sla_target_ms:
        print(f"\n❌ SLA BREACH: p95 {p95:.0f}ms > {sla_target_ms}ms target")
        environment.process_exit_code = 1
    else:
        print(f"\n✅ SLA MET: p95 {p95:.0f}ms ≤ {sla_target_ms}ms target")
