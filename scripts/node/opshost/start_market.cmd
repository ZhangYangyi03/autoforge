@echo off
rem Launcher for the local toolmarket shelf (:8000), kept next to the node's.
rem Logs are appended, not thrown away: the first arrival probe was missed
rem because the process that served it had stdout discarded, and "not in the log
rem I read" then read as "did not happen" -- on both machines.
set PATH=C:\Users\china\miniconda3;C:\Users\china\miniconda3\Scripts;C:\Windows\System32;C:\Windows
rem LAN-visible on purpose: the peer machine reads this shelf directly.
rem The tunneled door is the node on 8077 (/market), which keeps its token.
set TOOLMARKET_HOST=0.0.0.0
cd /d "C:\Users\china"
"C:\Users\china\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe" "C:\Users\china\toolmarket_server.py" >> "C:\Users\china\toolmarket_server.out.log" 2>> "C:\Users\china\toolmarket_server.err.log"
