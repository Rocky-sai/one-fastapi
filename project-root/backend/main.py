"""
AI Virtual Coding Platform Backend
FastAPI server for code execution and AI assistance
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
import subprocess
import tempfile
import os
import sys
import traceback
from datetime import datetime
import logging
import uvicorn
import requests
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure AI API from environment variable
API_KEY = os.getenv("GEMINI_API_KEY")  # Now supports both Gemini and Perplexity
if not API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables!")
    raise ValueError("GEMINI_API_KEY must be set in .env file")

logger.info(f"Loaded API key: {API_KEY[:10]}...")

# Detect API type based on key format
if API_KEY.startswith("pplx-"):
    AI_PROVIDER = "perplexity"
    logger.info("Using Perplexity AI provider")
else:
    AI_PROVIDER = "gemini"
    logger.info("Using Gemini AI provider")
    import google.generativeai as genai
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize FastAPI app
app = FastAPI(
    title="AI Virtual Coding Platform API",
    description="Backend API for AI-powered code execution and assistance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (frontend)
frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
frontend_path = os.path.abspath(frontend_path)
print(f"Frontend path: {frontend_path}")

if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Serve individual static files
@app.get("/style.css")
async def serve_css():
    css_path = os.path.join(frontend_path, "style.css")
    print(f"CSS path: {css_path}, exists: {os.path.exists(css_path)}")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    else:
        raise HTTPException(status_code=404, detail="CSS file not found")

@app.get("/app.js")
async def serve_js():
    js_path = os.path.join(frontend_path, "app.js")
    print(f"JS path: {js_path}, exists: {os.path.exists(js_path)}")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    else:
        raise HTTPException(status_code=404, detail="JS file not found")

# Serve frontend at root path
@app.get("/app")
async def serve_frontend():
    html_path = os.path.join(frontend_path, "index.html")
    print(f"HTML path: {html_path}, exists: {os.path.exists(html_path)}")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        raise HTTPException(status_code=404, detail="HTML file not found")

# Pydantic models for request/response
class CodeExecutionRequest(BaseModel):
    code: str
    language: str
    input_data: Optional[str] = ""

class CodeExecutionResponse(BaseModel):
    output: str
    error: str
    execution_time: float
    success: bool

class AIAssistRequest(BaseModel):
    message: str
    code: Optional[str] = ""
    language: Optional[str] = "python"
    action: str  # "chat", "explain", "debug", "optimize", "autofill"

class AIAssistResponse(BaseModel):
    response: str
    code_suggestion: Optional[str] = None
    success: bool

class ConnectionManager:
    """WebSocket connection manager for real-time features"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

# Initialize connection manager
manager = ConnectionManager()

def reload_gemini_api_key():
    """Reload API key from environment file"""
    global API_KEY, AI_PROVIDER, gemini_model
    
    # Reload .env file
    load_dotenv(override=True)
    
    new_api_key = os.getenv("GEMINI_API_KEY")
    if new_api_key and new_api_key != API_KEY:
        API_KEY = new_api_key
        
        # Detect API type and configure accordingly
        if API_KEY.startswith("pplx-"):
            AI_PROVIDER = "perplexity"
            logger.info(f"🔄 Reloaded Perplexity API key: {API_KEY[:10]}...")
        else:
            AI_PROVIDER = "gemini"
            if 'genai' in globals():
                genai.configure(api_key=API_KEY)
                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            logger.info(f"🔄 Reloaded Gemini API key: {API_KEY[:10]}...")
        
        return True
    return False

async def reload_api_key_async():
    """Async wrapper for reload_gemini_api_key"""
    return reload_gemini_api_key()

async def call_perplexity_api(prompt: str) -> str:
    """Call Perplexity API for AI responses"""
    try:
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful coding assistant. Provide clear, practical programming advice."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 800,
            "temperature": 0.2
        }
        
        # Use requests for synchronous call, run in executor for async
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post("https://api.perplexity.ai/chat/completions", 
                                headers=headers, json=data, timeout=30)
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"Perplexity API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Perplexity API call failed: {str(e)}")
        return None

def clean_markdown_formatting(text: str) -> str:
    """Clean up markdown formatting for better display in the frontend"""
    import re
    
    # Remove bold markdown
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    
    # Remove italic markdown
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove header markdown
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    
    # Clean up numbered lists to be more readable
    text = re.sub(r'^(\d+\.)\s*\*\*(.*?)\*\*:\s*', r'\1 \2: ', text, flags=re.MULTILINE)
    
    # Remove remaining ** symbols
    text = text.replace('**', '')
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up line breaks
    text = text.replace('\n\n\n', '\n\n')
    
    return text.strip()

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "AI Virtual Coding Platform Backend",
        "version": "1.0.0",
        "status": "running",
        "frontend": "Visit http://localhost:8000/app for the web interface",
        "endpoints": {
            "frontend": "/app",
            "docs": "/docs",
            "execute": "/execute",
            "ai_assist": "/ai_assist",
            "health": "/health"
        }
    }

@app.post("/test_format_sample")
async def test_format_sample():
    """Test endpoint to capture a sample of code formatting"""
    # Simulate what we get from Gemini AI
    sample_ai_response = """```python
def factorial(n): return 1 if n <= 1 else n * factorial(n-1)
def fibonacci(n): a, b = 0, 1; [a, b for a, b in [(b, a+b) for _ in range(n)]][-1]; return a
class Calculator: def __init__(self): self.history = []; def add(self, a, b): result = a + b; self.history.append(f"{a} + {b} = {result}"); return result; def get_history(self): return self.history
numbers = [1, 2, 3, 4, 5]; squared = [x**2 for x in numbers]; print("Squared numbers:", squared)
if True: print("This is a test"); for i in range(3): print(f"Number: {i}"); if i == 2: print("Last number")
```"""
    
    logger.info("🧪 Testing format sample:")
    logger.info(f"🧪 Sample AI response length: {len(sample_ai_response)} characters")
    
    # Extract code from the sample
    extracted_code = extract_code_from_response(sample_ai_response, "python")
    logger.info(f"🧪 Extracted code length: {len(extracted_code)} characters")
    logger.info(f"🧪 Extracted code preview: {repr(extracted_code[:200])}")
    
    # Format the code
    formatted_code = format_python_code(extracted_code)
    logger.info(f"🧪 Formatted code length: {len(formatted_code)} characters")
    logger.info(f"🧪 Formatted code lines: {len(formatted_code.split(chr(10)))}")
    
    # Return both versions for comparison
    return {
        "raw_response": sample_ai_response,
        "extracted_code": extracted_code,
        "formatted_code": formatted_code,
        "extracted_lines": len(extracted_code.split('\n')),
        "formatted_lines": len(formatted_code.split('\n'))
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": "running",
        "gemini_api_key": f"{API_KEY[:10]}..." if API_KEY else "Not configured"
    }

@app.post("/reload-api-key")
async def reload_api_key():
    """Reload Gemini API key from .env file without restarting server"""
    try:
        if reload_gemini_api_key():
            return {
                "status": "success",
                "message": "API key reloaded successfully",
                "new_key_preview": f"{API_KEY[:10]}...",
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "status": "no_change",
                "message": "API key unchanged or not found in .env file",
                "current_key_preview": f"{API_KEY[:10]}..." if API_KEY else "Not configured",
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Failed to reload API key: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to reload API key: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def execute_python_code(code: str, input_data: str = "") -> tuple:
    """Execute Python code safely in a temporary file"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        # Create process with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tempfile.gettempdir()
        )
        
        # Execute with timeout
        try:
            stdout, stderr = process.communicate(input=input_data, timeout=10)
            return stdout, stderr, process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            return "", "Execution timeout (10 seconds)", 1
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
                
    except Exception as e:
        return "", f"Execution error: {str(e)}", 1

def execute_javascript_code(code: str, input_data: str = "") -> tuple:
    """Execute JavaScript code using Node.js"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            # Wrap code to handle input
            wrapped_code = f"""
const readline = require('readline');
const rl = readline.createInterface({{
    input: process.stdin,
    output: process.stdout
}});

// Mock console.input for input handling
global.input = function() {{
    return new Promise((resolve) => {{
        rl.question('', (answer) => {{
            resolve(answer);
        }});
    }});
}};

// User code
{code}

rl.close();
"""
            f.write(wrapped_code)
            temp_file = f.name
        
        # Check if Node.js is available
        try:
            subprocess.run(['node', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "", "Node.js not found. Please install Node.js to run JavaScript code.", 1
        
        # Execute JavaScript
        process = subprocess.Popen(
            ['node', temp_file],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = process.communicate(input=input_data, timeout=10)
            return stdout, stderr, process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            return "", "Execution timeout (10 seconds)", 1
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
                
    except Exception as e:
        return "", f"Execution error: {str(e)}", 1

def simulate_other_languages(code: str, language: str) -> tuple:
    """Simulate execution for languages without runtime"""
    simulated_output = f"""
🔧 {language.upper()} Code Analysis:
{'=' * 40}

✅ Syntax appears valid
✅ Code structure looks good
🔍 Code length: {len(code)} characters
📝 Lines of code: {len(code.splitlines())}

Note: This is a simulation. To execute {language} code, 
please install the appropriate compiler/interpreter:

Python: Already supported
JavaScript: Install Node.js
Java: Install JDK and javac
C++: Install g++ compiler
Go: Install Go compiler

Your code:
{'-' * 20}
{code[:500]}{'...' if len(code) > 500 else ''}
"""
    return simulated_output, "", 0

@app.post("/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    """Execute code in the specified language"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f"Executing {request.language} code")
        
        if request.language.lower() == "python":
            stdout, stderr, returncode = execute_python_code(request.code, request.input_data)
        elif request.language.lower() == "javascript":
            stdout, stderr, returncode = execute_javascript_code(request.code, request.input_data)
        else:
            # Simulate execution for other languages
            stdout, stderr, returncode = simulate_other_languages(request.code, request.language)
        
        execution_time = asyncio.get_event_loop().time() - start_time
        success = returncode == 0
        
        # Combine output and error
        output = stdout if stdout else ""
        error = stderr if stderr else ""
        
        if not success and not error:
            error = "Code execution failed"
        
        return CodeExecutionResponse(
            output=output,
            error=error,
            execution_time=execution_time,
            success=success
        )
        
    except Exception as e:
        execution_time = asyncio.get_event_loop().time() - start_time
        logger.error(f"Execution failed: {str(e)}")
        return CodeExecutionResponse(
            output="",
            error=f"Internal error: {str(e)}",
            execution_time=execution_time,
            success=False
        )

async def handle_perplexity_request(message: str, code: str, language: str, action: str) -> dict:
    """Handle AI requests using Perplexity API"""
    try:
        # Create appropriate prompt based on action
        if action == "chat":
            prompt = f"You are a coding assistant. User asks: {message}\n\nProvide a helpful response in plain text."
        elif action == "explain":
            if not code.strip():
                return {"response": "Please provide some code for me to explain!", "code_suggestion": None}
            prompt = f"Explain this {language} code clearly:\n\n{code}\n\nStructure your response as: 1) What it does 2) How it works 3) Key concepts"
        elif action == "debug":
            if not code.strip():
                return {"response": "Please provide some code for me to debug!", "code_suggestion": None}
            prompt = f"Debug this {language} code and identify issues:\n\n{code}\n\nProvide: 1) Syntax issues 2) Logic errors 3) Quick fixes"
        elif action == "optimize":
            if not code.strip():
                return {"response": "Please provide some code for me to optimize!", "code_suggestion": None}
            prompt = f"Optimize this {language} code:\n\n{code}\n\nProvide: 1) Analysis 2) Improvements 3) Optimized version"
        elif action == "autofill":
            if message.strip():
                prompt = f"Generate complete {language} code for: {message}\n\nProvide working code with comments in a code block using ```{language}"
            else:
                prompt = f"Generate a {language} template with comments in a code block using ```{language}"
        else:
            prompt = f"You are a helpful coding assistant. User asks: {message}"
        
        # Call Perplexity API
        ai_response = await call_perplexity_api(prompt)
        
        if ai_response:
            # Clean response
            ai_response = clean_markdown_formatting(ai_response)
            
            # Extract code for relevant actions
            code_suggestion = None
            if action in ["optimize", "autofill"]:
                code_suggestion = extract_code_from_response(ai_response, language)
            
            return {
                "response": ai_response,
                "code_suggestion": code_suggestion
            }
        else:
            return {"response": "Perplexity AI is not responding. Please try again.", "code_suggestion": None}
            
    except Exception as e:
        logger.error(f"Perplexity request failed: {str(e)}")
        return {"response": f"Perplexity AI error: {str(e)}", "code_suggestion": None}

def format_python_code(code: str) -> str:
    """Format Python code to ensure proper 4-space indentation"""
    logger.info(f"🔧 Formatting Python code: {len(code)} characters")
    
    if not code.strip():
        return code
    
    # Debug: Show first 200 chars of input with special character visibility
    logger.info(f"🔧 Input preview: {repr(code[:200])}")
    logger.info(f"🔧 Has \\n: {chr(10) in code}, Has \\r: {chr(13) in code}")
    logger.info(f"🔧 Split by \\n gives: {len(code.split(chr(10)))} parts")
    logger.info(f"🔧 Split by \\r\\n gives: {len(code.split(chr(13)+chr(10)))} parts")
    
    # Normalize line endings first
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    
    # Count actual lines after normalization
    initial_lines = code.split('\n')
    logger.info(f"🔧 After normalization: {len(initial_lines)} lines")
    
    # If still single line, force aggressive splitting
    if len(initial_lines) <= 1 and len(code) > 100:
        logger.info("🔧 Single line detected after normalization, forcing split...")
        import re
        
        # Strategy 1: Split on Python keywords that should be on new lines
        keywords = ['def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'else:', 'elif ', 'finally:', 'import ', 'from ', 'return ', 'print(']
        
        for keyword in keywords:
            if keyword in code:
                # Replace with newline + keyword (but not at start)
                code = re.sub(f'(?<!^){re.escape(keyword)}', f'\n{keyword}', code)
                logger.info(f"🔧 Split on '{keyword}', now {len(code.split(chr(10)))} lines")
        
        # Strategy 2: Split after colons (Python block structure)
        code = re.sub(r':(\s*)([a-zA-Z_])', r':\n    \2', code)
        
        # Strategy 3: Split on semicolons if any
        if ';' in code:
            code = code.replace(';', ';\n')
        
        # Clean up excessive newlines
        code = re.sub(r'\n+', '\n', code).strip()
        
        final_lines = code.split('\n')
        logger.info(f"🔧 After forced splitting: {len(final_lines)} lines")
        logger.info(f"🔧 Force split preview: {repr(code[:300])}")
    
    lines = code.split('\n')
    formatted_lines = []
    indent_level = 0
    
    logger.info(f"🔧 Processing {len(lines)} lines for indentation")
    
    for line_num, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            formatted_lines.append('')
            continue
            
        # Handle comments - preserve but apply proper indentation
        if stripped.startswith('#'):
            formatted_lines.append('    ' * indent_level + stripped)
            continue
            
        # Handle dedent keywords (decrease indent before processing)
        if stripped.startswith(('else:', 'elif ', 'except', 'except:', 'finally:')):
            indent_level = max(0, indent_level - 1)
        
        # Apply current indentation to the line
        formatted_line = '    ' * indent_level + stripped
        formatted_lines.append(formatted_line)
        
        # Handle indent keywords (increase indent after processing)
        if stripped.endswith(':') and any(stripped.startswith(kw) for kw in [
            'if ', 'for ', 'while ', 'try:', 'def ', 'class ', 'with ', 'else:', 'elif ', 'except', 'finally:'
        ]):
            indent_level += 1
    
    result = '\n'.join(formatted_lines)
    logger.info(f"🔧 Formatted result: {len(result)} characters, {len(formatted_lines)} lines")
    return result
    
    for line_num, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            formatted_lines.append('')
            continue
            
        # Handle comments - preserve but apply proper indentation
        if stripped.startswith('#'):
            formatted_lines.append('    ' * indent_level + stripped)
            continue
            
        # Handle dedent keywords (decrease indent before processing)
        if stripped.startswith(('else:', 'elif ', 'except', 'except:', 'finally:')):
            indent_level = max(0, indent_level - 1)
        
        # Apply current indentation to the line
        formatted_line = '    ' * indent_level + stripped
        formatted_lines.append(formatted_line)
        
        # Handle indent keywords (increase indent after processing)
        if stripped.endswith(':') and any(stripped.startswith(kw) for kw in [
            'if ', 'for ', 'while ', 'try:', 'def ', 'class ', 'with ', 'else:', 'elif ', 'except', 'finally:'
        ]):
            indent_level += 1
    
    result = '\n'.join(formatted_lines)
    logger.info(f"🔧 Formatted result: {len(result)} characters, {len(formatted_lines)} lines")
    return result

def extract_code_from_response(ai_response: str, language: str) -> str:
    """Extract code blocks from AI response"""
    import re
    
    # Try to find code blocks with language specification first
    pattern = rf'```{language}\s*\n(.*?)\n```'
    matches = re.findall(pattern, ai_response, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        # Try to find any code blocks
        pattern = r'```\w*\s*\n(.*?)\n```'
        matches = re.findall(pattern, ai_response, re.DOTALL)
    
    if not matches:
        # Try without language specification
        pattern = r'```(.*?)```'
        matches = re.findall(pattern, ai_response, re.DOTALL)
    
    if matches:
        # Use the longest code block
        code_suggestion = max(matches, key=len).strip()
        
        # Clean up the extracted code
        lines = code_suggestion.split('\n')
        cleaned_lines = []
        for line in lines:
            if not cleaned_lines and not line.strip():
                continue
            cleaned_lines.append(line)
        
        # Remove trailing empty lines
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        if cleaned_lines:
            final_code = '\n'.join(cleaned_lines)
            
            # Apply Python-specific formatting if it's Python code
            if language.lower() == 'python':
                final_code = format_python_code(final_code)
            
            return final_code
    
    return None

async def get_ai_response(message: str, code: str, language: str, action: str) -> dict:
    """Generate AI responses using either Gemini or Perplexity AI"""
    
    # Check if we're using Perplexity API
    if AI_PROVIDER == "perplexity":
        return await handle_perplexity_request(message, code, language, action)
    
    # Continue with Gemini logic for backward compatibility
    try:
        # Create context-aware prompts based on action
        if action == "chat":
            prompt = f"""You are an expert coding assistant. The user is asking: "{message}"

Provide a helpful, concise response about programming or development topics.
Rules:
- Keep responses under 200 words for simple questions
- For complex topics, provide structured explanations
- Use plain text formatting (no markdown symbols like ** or ##)
- Be direct and practical
- Give examples when helpful

Respond in plain text without markdown formatting."""

        elif action == "explain":
            if not code.strip():
                return {"response": "Please provide some code for me to explain!", "code_suggestion": None}
            
            # Determine response length based on code complexity
            code_lines = len(code.strip().split('\n'))
            if code_lines <= 3:
                response_type = "brief"
            elif code_lines <= 10:
                response_type = "medium"
            else:
                response_type = "detailed"
            
            if response_type == "brief":
                prompt = f"""Explain this simple {language} code briefly and clearly:

{code}

Provide a concise explanation in 2-3 sentences without markdown formatting.
Focus on what it does and how it works in plain English."""
            
            elif response_type == "medium":
                prompt = f"""Explain this {language} code clearly:

{code}

Provide a structured explanation:
1. What this code does (1-2 sentences)
2. How it works step by step
3. Key concepts used

Use plain text formatting, no markdown symbols."""
            
            else:
                prompt = f"""Provide a comprehensive explanation of this {language} code:

{code}

Structure your response as:
1. Overview: What does this code do overall?
2. Step-by-step breakdown: Explain each major section
3. Key concepts: Important programming concepts used
4. How it works: The logic and flow of execution
5. Potential improvements: Brief suggestions

Use plain text formatting, no markdown symbols like ** or ##."""

        elif action == "debug":
            if not code.strip():
                return {"response": "Please provide some code for me to debug!", "code_suggestion": None}
            
            # Analyze code complexity for appropriate response
            code_lines = len(code.strip().split('\n'))
            has_functions = 'def ' in code or 'function ' in code
            has_classes = 'class ' in code
            
            if code_lines <= 5 and not has_functions and not has_classes:
                prompt = f"""Analyze this simple {language} code for any issues:

{code}

Provide a brief analysis focusing on:
- Any obvious errors or issues
- Quick improvements if needed
- Overall assessment (good/needs fixes)

Keep response concise, use plain text, no markdown formatting."""
            
            else:
                prompt = f"""Analyze this {language} code for bugs and issues:

{code}

Provide a structured debugging analysis:
1. Syntax Issues: Any syntax errors or problems
2. Logic Errors: Potential runtime issues or logical mistakes  
3. Code Quality: Improvements for better readability
4. Quick Fixes: Most important issues to address first

Use plain text formatting, avoid markdown symbols like ** or ##."""

        elif action == "optimize":
            if not code.strip():
                return {"response": "Please provide some code for me to optimize!", "code_suggestion": None}
            
            code_lines = len(code.strip().split('\n'))
            
            if code_lines <= 5:
                prompt = f"""Analyze this simple {language} code for optimization:

{code}

Provide brief optimization suggestions:
- Current approach assessment
- Key improvements possible
- Optimized version if beneficial

Keep response focused and practical, use plain text formatting."""
            
            else:
                prompt = f"""Optimize this {language} code for better performance:

{code}

Provide optimization analysis:
1. Current Analysis: What the code does and current approach
2. Performance Issues: Main bottlenecks or inefficiencies
3. Optimization Strategies: Specific improvements possible
4. Optimized Code: Improved version with better structure
5. Benefits: What improvements were made and why

Use plain text formatting, avoid markdown symbols."""

        elif action == "autofill":
            if message.strip():
                # User provided specific request
                if language == "python":
                    prompt = f"""Generate complete, working Python code for: "{message}"

CRITICAL FORMATTING REQUIREMENTS:
- Use EXACTLY 4 spaces for each indentation level (NO TABS)
- Every function body must be indented 4 spaces from the def line
- Every class body must be indented 4 spaces from the class line
- Every if/for/while/try block must be indented 4 spaces
- Comments should be properly aligned
- No mixing of tabs and spaces

Format your response EXACTLY like this:
1. Start with a brief explanation (1-2 sentences)
2. Then provide the complete code wrapped in triple backticks:

```python
# Complete working Python code here
# Each line properly indented with 4 spaces
# Functions properly defined with 4-space indentation
def example_function():
    # Function body indented 4 spaces
    print("This is properly indented")
    return True

# Main execution
if __name__ == "__main__":
    # Main code indented 4 spaces
    result = example_function()
    print(f"Result: {{result}}")
```

Requirements for the code:
- Complete and executable without external dependencies
- Include sample data or test cases within the code
- Add clear comments explaining each section
- Follow PEP 8 Python style guide with 4-space indentation
- Make it educational and self-contained
- Ensure ALL code blocks are properly indented

Example: For "sum of two numbers", provide a complete program with proper 4-space indentation throughout."""
                else:
                    prompt = f"""Generate complete, working {language} code for: "{message}"

IMPORTANT: You must provide the code in a structured format with proper code blocks.

Format your response EXACTLY like this:
1. Start with a brief explanation (1-2 sentences)
2. Then provide the complete code wrapped in triple backticks like this:

```{language}
# Your complete working code here
# Include comments explaining each part
# Make sure the code is ready to run
```

Requirements for the code:
- Complete and executable without external dependencies
- Include sample data or test cases within the code
- Add clear comments explaining each section
- Follow {language} best practices
- Make it educational and self-contained

Example: For "sum of two numbers", provide a complete program that takes input and calculates sum."""
            else:
                # Default template
                if language == "python":
                    prompt = f"""Generate a comprehensive Python code template with proper formatting and strict 4-space indentation.

CRITICAL INDENTATION RULES:
- Use EXACTLY 4 spaces for indentation (NO TABS)
- Every function body indented 4 spaces from def
- Every class body indented 4 spaces from class
- Every if/for/while/try block indented 4 spaces
- Nested blocks use 8 spaces (2 levels of 4-space indentation)

Format your response EXACTLY like this:
1. Brief explanation of what the template demonstrates
2. Complete code in proper code block format:

```python
# Complete working Python template here
# Every line properly indented with 4 spaces
def main_function():
    # Function body indented exactly 4 spaces
    print("Proper Python indentation example")
    
    # Nested structures use 8 spaces
    for i in range(3):
        print(f"Iteration {{i}}")
        if i > 0:
            print("    Nested condition")
    
    return "Template complete"

# Main execution block
if __name__ == "__main__":
    # Main code indented 4 spaces
    result = main_function()
    print(f"Result: {{result}}")
```

Make the code educational, executable, and demonstrate proper Python formatting with strict 4-space indentation."""
                else:
                    prompt = f"""Generate a comprehensive {language} code template with proper formatting.

Format your response EXACTLY like this:
1. Brief explanation of what the template demonstrates
2. Complete code in proper code block format:

```{language}
# Complete working code template here
# Include comments and explanations
# Show essential {language} patterns
```

Make the code educational, executable, and well-documented."""

        else:
            # Default chat
            prompt = f"""You are a helpful coding assistant. The user is asking: "{message}"
            
Provide a helpful, concise response under 200 words."""

        # Generate response using Gemini with timeout and retry logic
        logger.info(f"Sending request to Gemini AI for action: {action}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Create generation config for better structured responses
        if action == "autofill":
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=1000,  # More tokens for complete code examples
                temperature=0.1,         # Very low temperature for consistent formatting
            )
        else:
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=700,   # More tokens for detailed responses
                temperature=0.2,         # Low temperature for focused responses
            )
        
        # Retry logic for quota issues
        for attempt in range(3):
            try:
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, 
                    lambda: gemini_model.generate_content(prompt, generation_config=generation_config)
                )
                
                logger.info(f"Received response from Gemini AI on attempt {attempt + 1}")
                
                if response and response.text:
                    ai_response = response.text.strip()
                    logger.info(f"Raw AI response length: {len(ai_response)} characters")
                    
                    # Clean up markdown formatting for better display
                    ai_response = clean_markdown_formatting(ai_response)
                    
                    # Extract code suggestions for optimize and autofill actions
                    code_suggestion = None
                    if action in ["optimize", "autofill"]:
                        # Look for code blocks in the response
                        import re
                        
                        # Try to find code blocks with language specification first (most reliable)
                        pattern = rf'```{language}\s*\n(.*?)\n```'
                        matches = re.findall(pattern, ai_response, re.DOTALL | re.IGNORECASE)
                        
                        if not matches:
                            # Try to find any code blocks with any language
                            pattern = r'```\w*\s*\n(.*?)\n```'
                            matches = re.findall(pattern, ai_response, re.DOTALL)
                        
                        if not matches:
                            # Try to find code blocks without language specification
                            pattern = r'```(.*?)```'
                            matches = re.findall(pattern, ai_response, re.DOTALL)
                        
                        if matches:
                            # Use the longest code block (most likely to be the main code)
                            code_suggestion = max(matches, key=len).strip()
                            
                            # Clean up the extracted code
                            lines = code_suggestion.split('\n')
                            cleaned_lines = []
                            for line in lines:
                                # Skip empty lines at the beginning
                                if not cleaned_lines and not line.strip():
                                    continue
                                cleaned_lines.append(line)
                            
                            # Remove trailing empty lines
                            while cleaned_lines and not cleaned_lines[-1].strip():
                                cleaned_lines.pop()
                            
                            if cleaned_lines:
                                code_suggestion = '\n'.join(cleaned_lines)
                                logger.info(f"🔍 Extracted code before formatting: {len(code_suggestion)} chars")
                                
                                # Apply Python-specific formatting if it's Python code
                                if language.lower() == 'python':
                                    code_suggestion = format_python_code(code_suggestion)
                                    logger.info(f"🔍 Formatted Python code: {len(code_suggestion)} chars")
                        
                        # For autofill, if still no code found, try alternative extraction
                        if action == "autofill" and not code_suggestion:
                            # Look for code-like patterns in the response
                            lines = ai_response.split('\n')
                            code_lines = []
                            in_code_section = False
                            
                            for line in lines:
                                stripped = line.strip()
                                # Start collecting when we see code-like patterns
                                if (stripped.startswith(('def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ', 'try:', 'function ', 'const ', 'let ', 'var ', 'public ', 'private ', '#', '//', 'print(', 'console.log')) or
                                    '=' in stripped and not stripped.startswith(('1.', '2.', '3.', '-', '*')) or
                                    stripped.endswith((':',';', '{', '}')) or
                                    stripped.startswith(('    ', '\t'))):
                                    in_code_section = True
                                    code_lines.append(line)
                                elif in_code_section:
                                    # Continue if it's an empty line or looks like code
                                    if not stripped or stripped.startswith(('    ', '\t', '#', '//')):
                                        code_lines.append(line)
                                    elif any(keyword in stripped for keyword in ['def ', 'class ', 'if ', 'for ', 'while ', 'print(', '=']):
                                        code_lines.append(line)
                                    else:
                                        # Stop if we hit explanatory text
                                        if len(code_lines) > 2:  # Only stop if we have substantial code
                                            break
                            
                            if code_lines and len(code_lines) >= 2:
                                # Clean up the extracted code
                                while code_lines and not code_lines[0].strip():
                                    code_lines.pop(0)
                                while code_lines and not code_lines[-1].strip():
                                    code_lines.pop()
                                
                                if code_lines:
                                    code_suggestion = '\n'.join(code_lines)
                                    logger.info(f"🔍 Alternative extraction before formatting: {len(code_suggestion)} chars")
                                    
                                    # Apply Python-specific formatting if it's Python code
                                    if language.lower() == 'python':
                                        code_suggestion = format_python_code(code_suggestion)
                                        logger.info(f"🔍 Alternative extraction after formatting: {len(code_suggestion)} chars")
                    
                    # Final formatting check - ensure any code_suggestion for Python gets formatted
                    if language.lower() == 'python' and code_suggestion:
                        logger.info(f"🔧 Final Python formatting check: {len(code_suggestion)} chars")
                        code_suggestion = format_python_code(code_suggestion)
                    
                    return {
                        "response": ai_response,
                        "code_suggestion": code_suggestion
                    }
                else:
                    logger.warning(f"Empty response from Gemini AI on attempt {attempt + 1}")
                    if attempt < 2:  # Don't wait on last attempt
                        await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
                        continue
                    else:
                        return {"response": "Gemini AI returned an empty response. Please try again.", "code_suggestion": None}
                    
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Gemini AI error on attempt {attempt + 1}: {error_msg}")
                
                # Check for quota exceeded
                if "429" in error_msg or "quota" in error_msg.lower():
                    logger.warning("Quota exceeded, attempting to reload API key...")
                    
                    # Try to reload API key from .env file
                    if reload_gemini_api_key():
                        logger.info("API key reloaded, retrying request...")
                        # Don't return error, let it retry with new key
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                    
                    return {
                        "response": f"⚠️ {AI_PROVIDER.capitalize()} AI quota exceeded for current API key. Please update GEMINI_API_KEY in your .env file with a new key, then use the 'Reload API Key' button or restart the server. Current key: {API_KEY[:10]}...",
                        "code_suggestion": None
                    }
                
                # Check for other API errors
                if "40" in error_msg:  # 400-level errors
                    return {
                        "response": f"Gemini AI API error: {error_msg}. Please check your request and try again.",
                        "code_suggestion": None
                    }
                
                # For other errors, retry with backoff
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    return {
                        "response": f"Gemini AI error after {attempt + 1} attempts: {error_msg}. Please try again later.",
                        "code_suggestion": None
                    }
        
        # If all retries failed
        return {
            "response": "Unable to get response from Gemini AI after multiple attempts. Please try again later.",
            "code_suggestion": None
        }
            
    except asyncio.TimeoutError:
        logger.error("Gemini AI request timed out - retrying with shorter prompt")
        # Retry with a simpler prompt instead of fallback
        simple_prompt = f"Generate {language} code for: {message}. Provide working code in ```{language} code blocks."
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: gemini_model.generate_content(simple_prompt)
            )
            
            if response and response.text:
                ai_response = clean_markdown_formatting(response.text.strip())
                
                # Extract code from retry response
                code_suggestion = None
                if action in ["optimize", "autofill"]:
                    import re
                    pattern = rf'```{language}\s*\n(.*?)\n```'
                    matches = re.findall(pattern, ai_response, re.DOTALL | re.IGNORECASE)
                    if matches:
                        code_suggestion = max(matches, key=len).strip()
                
                return {"response": ai_response, "code_suggestion": code_suggestion}
            else:
                return {"response": "Unable to get response from Gemini AI. Please try again.", "code_suggestion": None}
                
        except Exception:
            return {"response": "Gemini AI is temporarily unavailable. Please try again in a moment.", "code_suggestion": None}
            
    except Exception as e:
        logger.error(f"Gemini AI error: {str(e)} - retrying...")
        
        # Retry with basic prompt instead of using fallback
        try:
            basic_prompt = f"Create {language} code for: {message}"
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: gemini_model.generate_content(basic_prompt)
            )
            
            if response and response.text:
                ai_response = clean_markdown_formatting(response.text.strip())
                
                # Extract code from retry response
                code_suggestion = None
                if action in ["optimize", "autofill"]:
                    import re
                    pattern = rf'```{language}\s*\n(.*?)\n```'
                    matches = re.findall(pattern, ai_response, re.DOTALL | re.IGNORECASE)
                    if not matches:
                        pattern = r'```(.*?)```'
                        matches = re.findall(pattern, ai_response, re.DOTALL)
                    if matches:
                        code_suggestion = max(matches, key=len).strip()
                        
                        # Apply Python formatting if it's Python code
                        if language.lower() == 'python':
                            logger.info(f"🔧 Retry path Python formatting: {len(code_suggestion)} chars")
                            code_suggestion = format_python_code(code_suggestion)
                
                return {"response": ai_response, "code_suggestion": code_suggestion}
            else:
                return {"response": "Gemini AI response was empty. Please try again with a different request.", "code_suggestion": None}
                
        except Exception as retry_error:
            logger.error(f"Gemini AI retry failed: {str(retry_error)}")
            return {"response": f"Gemini AI error: {str(e)}. Please check your internet connection and try again.", "code_suggestion": None}

@app.post("/ai_assist", response_model=AIAssistResponse)
async def ai_assist(request: AIAssistRequest):
    """AI assistance for coding tasks"""
    try:
        logger.info(f"AI assist request: {request.action}")
        
        # Add timeout to the AI response - increased for better Gemini results
        result = await asyncio.wait_for(
            get_ai_response(
                request.message, 
                request.code or "", 
                request.language, 
                request.action
            ),
            timeout=30.0  # Increased timeout to get real Gemini responses
        )
        
        return AIAssistResponse(
            response=result["response"],
            code_suggestion=result["code_suggestion"],
            success=True
        )
        
    except asyncio.TimeoutError:
        logger.error("AI assist request timed out")
        return AIAssistResponse(
            response="The request timed out. Please try again with a shorter message.",
            code_suggestion=None,
            success=False
        )
    except Exception as e:
        logger.error(f"AI assist failed: {str(e)}")
        return AIAssistResponse(
            response=f"Sorry, I encountered an error: {str(e)}",
            code_suggestion=None,
            success=False
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Echo message back (you can add more logic here)
            response = {
                "type": "message",
                "content": f"Received: {message_data.get('message', '')}",
                "timestamp": datetime.now().isoformat()
            }
            
            await manager.send_personal_message(json.dumps(response), websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "status_code": 404}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
