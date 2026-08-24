@echo off
title HalluciGuard Production Engine
echo ====================================================
echo Starting HalluciGuard Backend Engine on port 8000...
echo ====================================================
start "HalluciGuard Backend" python -m uvicorn orchestration.api:app --host 0.0.0.0 --port 8000
echo.
echo Starting Cloudflare Tunnel for Mobile & Remote access...
echo ====================================================
.\cloudflared.exe tunnel --url http://127.0.0.1:8000
pause
