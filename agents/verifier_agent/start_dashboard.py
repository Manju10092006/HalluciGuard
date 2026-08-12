import sys
import os
import uvicorn

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    print("======================================================================")
    print("Starting HalluciGuard Verifier Agent Dashboard on http://127.0.0.1:8002")
    print("======================================================================")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8002, reload=False)
