import sys
import os
from dotenv import load_dotenv

load_dotenv()

# Automatically add agents/verifier_agent to sys.path
verifier_dir = os.path.join(os.path.dirname(__file__), "agents", "verifier_agent")
if verifier_dir not in sys.path:
    sys.path.insert(0, verifier_dir)

if __name__ == "__main__":
    import uvicorn
    print("================================================================")
    print("  Starting HalluciGuard Verifier Agent Server")
    print("  Swagger UI : http://localhost:8000/docs")
    print("  Dashboard  : http://localhost:8000/")
    print("================================================================")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
