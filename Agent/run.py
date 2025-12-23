import uvicorn
import sys
from config import settings
import cli
from api import app


if __name__ == "__main__":
    mode = settings.EXECUTION_MODE.lower()
    
    print(f"Starting Agent 01 in '{mode}' mode...")
    
    if mode == "local":
        cli.main()
    elif mode == "server":
        uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
    else:
        print(f"Error: Invalid EXECUTION_MODE '{mode}'. Must be 'local' or 'server'.")
        sys.exit(1)
