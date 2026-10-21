@echo off
rem Start the ProcGen RL control room and open it in the browser.
rem Closing this window stops the web app; a running training keeps going
rem and is picked up again the next time the web app starts.
cd /d "%~dp0"
python webapp\server.py --open
pause
