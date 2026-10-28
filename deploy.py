from __future__ import annotations

import argparse
import asyncio
import os

import vertexai
from dotenv import load_dotenv
from vertexai import agent_engines

load_dotenv()

PROJECT = os.environ["GOOGLE_CLOUD_PROJECT"]
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
BUCKET = os.environ.get("STAGING_BUCKET")
MODEL = os.environ.get("DEFECT_AGENT_MODEL", "claude-opus-5-5")

REQUIREMENTS = [
    "google-adk>=2.13.0",
    "google-cloud-aiplatform[agent_engines,adk]>=2.6.0",
    "networkx>=3.3",
    "anthropic>=0.43.0",
]

SAMPLE_TURNS = [
    "Parts off press 7 have silver streaks near the gate and a few snapped during the drop test.",
    "No black specks. The moisture test just came back above the limit.",
]


def deploy() -> None:
    if not BUCKET:
        raise SystemExit("Set STAGING_BUCKET=gs://your-bucket")
    vertexai.init(project=PROJECT, location=LOCATION, staging_bucket=BUCKET)
    from defect_agent.agent import root_agent

    app = agent_engines.AdkApp(agent=root_agent, enable_tracing=True)
    remote = agent_engines.create(
        agent_engine=app,
        display_name="defect-diagnoser",
        description="Diagnoses injection-molding defects from a knowledge graph with calibrated confidence.",
        requirements=REQUIREMENTS,
        extra_packages=["./defect_agent"],          # ships kg.py and kg_data.json
        env_vars={"DEFECT_AGENT_MODEL": MODEL},
    )
    print("Deployed:", remote.resource_name)
    print(f"Try it:  python deploy.py query --resource {remote.resource_name}")


async def _query(resource: str) -> None:
    vertexai.init(project=PROJECT, location=LOCATION)
    remote = agent_engines.get(resource)
    session = await remote.async_create_session(user_id="qa-engineer")
    sid = session["id"] if isinstance(session, dict) else session.id
    for turn in SAMPLE_TURNS:
        print(f"\n>>> {turn}\n")
        async for event in remote.async_stream_query(user_id="qa-engineer", session_id=sid, message=turn):
            for part in event.get("content", {}).get("parts", []):
                if "text" in part:
                    print(part["text"], end="")
        print()


def delete(resource: str) -> None:
    vertexai.init(project=PROJECT, location=LOCATION)
    agent_engines.get(resource).delete(force=True)
    print("Deleted", resource)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["deploy", "query", "delete"])
    ap.add_argument("--resource", help="projects/.../reasoningEngines/... (for query/delete)")
    a = ap.parse_args()
    if a.action == "deploy":
        deploy()
    elif not a.resource:
        ap.error("--resource is required")
    elif a.action == "query":
        asyncio.run(_query(a.resource))
    else:
        delete(a.resource)
