#!/usr/bin/env python3
"""
Startup script for AI Virtual Coding Platform Backend
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        sys.exit(1)
    logger.info(f"Python version: {sys.version}")

def install_requirements():
    """Install required packages"""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        logger.info("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    else:
        logger.error("requirements.txt not found")
        sys.exit(1)

def check_dependencies():
    """Check if all dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import pydantic
        logger.info("All core dependencies are available")
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    logger.info("Starting AI Virtual Coding Platform Backend...")
    
    # Import and run the FastAPI app
    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    logger.info("=== AI Virtual Coding Platform Backend ===")
    
    # Check Python version
    check_python_version()
    
    # Install requirements if needed
    if not check_dependencies():
        logger.info("Installing missing dependencies...")
        install_requirements()
        
        # Check again after installation
        if not check_dependencies():
            logger.error("Failed to install required dependencies")
            sys.exit(1)
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()
