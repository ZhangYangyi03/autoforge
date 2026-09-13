"""Live AutoDL environment probe + the real-GPU half of the GPU layer's tests.

Runs on the box. Prints facts only; no assertions here.
"""
import json
import os
import shutil
import subprocess
import sys

def sh(cmd):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=120)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"

print("== python ==")
print("executable:", sys.executable)
print("version   :", sys.version.split()[0])

print("\n== PATH fix ==")
for p in ("/root/miniconda3/bin",):
    if p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + ":" + os.environ.get("PATH", "")
print("PATH head:", os.environ["PATH"].split(":")[:3])

print("\n== nvcc ==")
print("which nvcc:", shutil.which("nvcc") or "NONE")
print(sh("nvcc --version 2>&1 | tail -3").strip())

print("\n== torch ==")
print(sh("""python -c "
import torch
print('torch', torch.__version__)
print('cuda avail', torch.cuda.is_available())
print('cuda ver', torch.version.cuda)
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0)
    print('name', p.name)
    print('capability', p.major, p.minor)
    print('SMs', p.multi_processor_count)
    print('mem GB', round(p.total_memory/1e9,1))
" 2>&1""").strip())

print("\n== triton ==")
print(sh("""python -c "
import triton
print('triton', triton.__version__)
from triton.testing import do_bench
print('do_bench importable: yes')
" 2>&1""").strip())

print("\n== disks ==")
print(sh("df -h / /root/autodl-tmp 2>&1").strip())

print("\n== git in repo? ==")
print(sh("ls -d /root/labs 2>/dev/null || echo 'no /root/labs'").strip())
