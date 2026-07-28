import uvicorn

from .config.settings import get_settings

settings = get_settings()

if __name__ == "__main__":
    uvicorn.run(
        "agents.memory_agent.api.main:app",
        host="0.0.0.0",
        port=settings.memory_agent_port,
        reload=True,
    )
