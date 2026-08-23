@echo off
title WhatsApp AI Agent - Local Test
echo ==========================================
echo   WHATSAPP AI AGENT LOCAL SETUP
echo ==========================================
echo.
echo [1/2] Installing Dependencies...
pip install -r requirements.txt
echo.
echo [2/2] Starting Agent...
echo ------------------------------------------
echo IMPORTANT: A QR code will appear in this window. 
echo Scan it using WhatsApp > Linked Devices on your phone.
echo ------------------------------------------
python whatsapp_agent.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The bot crashed or stopped. Read the error message above.
)
pause
