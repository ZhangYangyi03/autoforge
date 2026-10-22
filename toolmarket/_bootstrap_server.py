import sys
sys.path.insert(0, r'D:\Users\china\Desktop\绿目_开发\autoforge')
sys.path.insert(0, r'D:\Users\china\Desktop\系目\开发\tool-market')
import uvicorn
uvicorn.run('toolmarket.api.main:app', host='127.0.0.1', port=8000)
