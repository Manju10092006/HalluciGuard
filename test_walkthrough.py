"""Live narrated tester for this checkout; existing test_direct.py is unchanged.

Run: python test_walkthrough.py "Who founded Microsoft?"
Actual production graph, LLM generation and retrieval are used. Conditional
agents are never forced. Reports and isolated Memory are saved in Downloads.
"""
from scripts.run_seven_agent_walkthrough import main


if __name__ == "__main__":
    raise SystemExit(main())
