@echo off
set PATH=C:\Users\china\miniconda3;C:\Users\china\miniconda3\Scripts;C:\Windows\System32;C:\Windows;C:\Program Files\OpenSSH
cd /d "C:\Users\china\autoforge_node"
"C:\Users\china\miniconda3\python.exe" "C:\Users\china\autoforge_node\node_server.py" --host 0.0.0.0 --port 8077 >> "C:\Users\china\autoforge_node\node.out.log" 2>> "C:\Users\china\autoforge_node\node.err.log"
