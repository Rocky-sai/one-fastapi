// AI Virtual Coding Platform JavaScript with Backend Integration

let editor;
let currentLanguage = 'python';
const BACKEND_URL = 'http://127.0.0.1:8000';

// Initialize Monaco Editor
function initializeEditor() {
  require.config({ paths: { 'vs': 'https://cdn.jsdelivr.net/npm/monaco-editor@0.34.1/min/vs' }});
  
  require(['vs/editor/editor.main'], function () {
    editor = monaco.editor.create(document.getElementById('editor'), {
      value: getDefaultCode('python'),
      language: 'python',
      theme: 'vs-dark',
      fontSize: 14,
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      automaticLayout: true,
      wordWrap: 'on',
      lineNumbers: 'on',
      renderWhitespace: 'selection',
      cursorStyle: 'line',
      formatOnPaste: true,
      formatOnType: true
    });

    // Update cursor position
    editor.onDidChangeCursorPosition((e) => {
      updateCursorPosition(e.position.lineNumber, e.position.column);
    });

    // Auto-save functionality
    editor.onDidChangeModelContent(() => {
      updateStatus('Modified');
    });
  });
}

// Get default code for each language
function getDefaultCode(language) {
  const defaultCode = {
    python: `# Welcome to AI Virtual Coding Platform
# Write your Python code here

def hello_world():
    print("Hello, World!")
    return "Welcome to Python coding!"

# Call the function
result = hello_world()
print(f"Result: {result}")`,

    javascript: `// Welcome to AI Virtual Coding Platform
// Write your JavaScript code here

function helloWorld() {
    console.log("Hello, World!");
    return "Welcome to JavaScript coding!";
}

// Call the function
const result = helloWorld();
console.log(\`Result: \${result}\`);`,

    java: `// Welcome to AI Virtual Coding Platform
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
        System.out.println("Welcome to Java coding!");
    }
}`,

    cpp: `// Welcome to AI Virtual Coding Platform
#include <iostream>
#include <string>

int main() {
    std::cout << "Hello, World!" << std::endl;
    std::cout << "Welcome to C++ coding!" << std::endl;
    return 0;
}`,

    html: `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My Web Page</title>
</head>
<body>
    <h1>Hello, World!</h1>
    <p>Welcome to HTML coding!</p>
</body>
</html>`,

    css: `/* Welcome to AI Virtual Coding Platform */
body {
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    text-align: center;
    padding: 50px;
}

h1 {
    font-size: 3rem;
    margin-bottom: 20px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}`
  };

  return defaultCode[language] || defaultCode.python;
}

// Update cursor position display
function updateCursorPosition(line, column) {
  document.getElementById('cursor-position').textContent = `Line ${line}, Column ${column}`;
}

// Update status
function updateStatus(status) {
  document.getElementById('status').textContent = status;
}

// Language change handler
function changeLanguage(language) {
  currentLanguage = language;
  if (editor) {
    const currentCode = editor.getValue();
    const isEmpty = !currentCode.trim() || currentCode === getDefaultCode(monaco.editor.getModel(editor.getModel().uri).getLanguageId());
    
    if (isEmpty) {
      editor.setValue(getDefaultCode(language));
    }
    
    monaco.editor.setModelLanguage(editor.getModel(), language === 'cpp' ? 'cpp' : language);
    updateStatus('Language changed to ' + language.toUpperCase());
  }
}

// Backend API calls
async function callBackendAPI(endpoint, data) {
  try {
    const response = await fetch(`${BACKEND_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Backend API call failed:', error);
    throw error;
  }
}

// Run code using backend
async function runCode() {
  const code = editor.getValue();
  const output = document.getElementById('output');
  
  if (!code.trim()) {
    output.textContent = 'Please write some code first!';
    return;
  }
  
  // Check if code requires input
  let inputData = "";
  if (code.includes('input(') && currentLanguage === 'python') {
    inputData = prompt("Your code uses input(). Please provide input values (one per line):");
    if (inputData === null) {
      output.textContent = 'Code execution cancelled.';
      return;
    }
  }
  
  updateStatus('Running...');
  output.textContent = 'Executing code...\n';
  
  try {
    const result = await callBackendAPI('/execute', {
      code: code,
      language: currentLanguage,
      input_data: inputData
    });
    
    let outputText = '';
    
    if (result.success) {
      outputText = `✅ Execution completed in ${result.execution_time.toFixed(3)}s\n\n`;
      outputText += result.output || 'No output produced';
      
      // Add helpful tips for different scenarios
      if (!result.output && code.trim()) {
        outputText += '\n\n💡 Tip: Make sure your code includes print() statements to see output.';
      }
    } else {
      outputText = `❌ Execution failed in ${result.execution_time.toFixed(3)}s\n\n`;
      outputText += result.error || 'Unknown error occurred';
      
      // Add helpful error suggestions
      if (result.error.includes('arguments are required')) {
        outputText += '\n\n💡 Tip: This code expects command line arguments. Try using autofill to generate self-contained code instead.';
      } else if (result.error.includes('No module named')) {
        outputText += '\n\n💡 Tip: This code requires external libraries. Use built-in Python modules or ask AI to generate code without external dependencies.';
      } else if (result.error.includes('SyntaxError')) {
        outputText += '\n\n💡 Tip: There\'s a syntax error in your code. Try using the Debug feature to find and fix issues.';
      }
    }
    
    output.textContent = outputText;
    updateStatus('Execution completed');
    
  } catch (error) {
    output.textContent = `❌ Backend connection failed:\n${error.message}\n\nMake sure the backend server is running at ${BACKEND_URL}`;
    updateStatus('Backend error');
  }
}

// Chat functionality with backend
function addChatMessage(message, isUser = false) {
  const chatContainer = document.getElementById('chat');
  const messageDiv = document.createElement('div');
  messageDiv.className = `chat-message ${isUser ? 'user' : 'ai'}`;
  messageDiv.textContent = message;
  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Send chat message to backend
async function sendChatMessage() {
  const input = document.getElementById('chat-input');
  const message = input.value.trim();
  
  if (message) {
    addChatMessage(message, true);
    input.value = '';
    
    // Show thinking indicator
    const thinkingDiv = document.createElement('div');
    thinkingDiv.className = 'chat-message ai thinking';
    thinkingDiv.innerHTML = '<div class="loading"></div> AI is thinking...';
    document.getElementById('chat').appendChild(thinkingDiv);
    
    try {
      const result = await callBackendAPI('/ai_assist', {
        message: message,
        code: editor.getValue(),
        language: currentLanguage,
        action: 'chat'
      });
      
      // Remove thinking indicator
      thinkingDiv.remove();
      
      if (result.success) {
        addChatMessage(result.response, false);
        
        // If there's a code suggestion, ask if user wants to apply it
        if (result.code_suggestion) {
          const applyButton = document.createElement('button');
          applyButton.textContent = 'Apply Code Suggestion';
          applyButton.className = 'btn btn-small btn-primary';
          applyButton.onclick = () => {
            editor.setValue(result.code_suggestion);
            applyButton.remove();
          };
          
          const messageDiv = document.createElement('div');
          messageDiv.className = 'chat-message ai';
          messageDiv.appendChild(applyButton);
          document.getElementById('chat').appendChild(messageDiv);
        }
      } else {
        addChatMessage('Sorry, I encountered an error processing your request.', false);
      }
      
    } catch (error) {
      thinkingDiv.remove();
      addChatMessage('Unable to connect to AI backend. Please check if the server is running.', false);
    }
  }
}

// AI action handlers with backend integration
async function performAIAction(action, customMessage = null) {
  const code = editor.getValue();
  
  if (!code.trim() && action !== 'autofill') {
    addChatMessage(`Please write some code first before using ${action}.`, false);
    return;
  }
  
  // Show loading message
  const loadingMessages = {
    'explain': '🔍 Analyzing your code...',
    'debug': '🐛 Searching for bugs and issues...',
    'optimize': '⚡ Finding optimization opportunities...',
    'autofill': '🤖 Generating code...'
  };
  
  addChatMessage(loadingMessages[action] || `Processing ${action} request...`, false);
  
  try {
    const result = await callBackendAPI('/ai_assist', {
      message: customMessage || `Please ${action} this code`,
      code: code,
      language: currentLanguage,
      action: action
    });
    
    if (result.success) {
      // Add the AI response
      addChatMessage(result.response, false);
      
      // If there's a code suggestion, offer to apply it
      if (result.code_suggestion && action !== 'explain') {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message ai action-buttons';
        
        const actionLabels = {
          'debug': 'Fixed Code',
          'optimize': 'Optimized Code',
          'autofill': 'Generated Code'
        };
        
        const applyButton = document.createElement('button');
        applyButton.textContent = `✅ Apply ${actionLabels[action] || 'Suggestion'}`;
        applyButton.className = 'btn btn-small btn-primary';
        applyButton.style.marginRight = '10px';
        applyButton.onclick = () => {
          editor.setValue(result.code_suggestion);
          messageDiv.remove();
          addChatMessage(`✅ Applied ${action} suggestion!`, false);
          updateStatus(`${action} applied`);
        };
        
        const previewButton = document.createElement('button');
        previewButton.textContent = '👀 Preview Changes';
        previewButton.className = 'btn btn-small btn-secondary';
        previewButton.onclick = () => {
          const previewDiv = document.createElement('div');
          previewDiv.className = 'code-preview';
          previewDiv.innerHTML = `<h4>Suggested ${actionLabels[action]}:</h4><pre><code>${escapeHtml(result.code_suggestion)}</code></pre>`;
          messageDiv.appendChild(previewDiv);
          previewButton.remove();
        };
        
        messageDiv.appendChild(applyButton);
        messageDiv.appendChild(previewButton);
        document.getElementById('chat').appendChild(messageDiv);
      }
    } else {
      addChatMessage(`Sorry, I couldn't ${action} your code. Please try again.`, false);
    }
    
  } catch (error) {
    addChatMessage(`Error connecting to AI service for ${action}. Please check your connection.`, false);
  }
}

async function autoFillCode() {
  // Ask user what they want to generate
  const userRequest = prompt("What code would you like me to generate? (e.g., 'a function to calculate fibonacci', 'a web scraper', 'a todo list app', etc.)");
  
  if (!userRequest) {
    addChatMessage("Autofill cancelled.", false);
    return;
  }
  
  // Show loading message
  addChatMessage(`🤖 Generating ${currentLanguage} code for: "${userRequest}"...`, false);
  
  try {
    const result = await callBackendAPI('/ai_assist', {
      message: userRequest,
      code: editor.getValue(),
      language: currentLanguage,
      action: 'autofill'
    });
    
    if (result.success) {
      addChatMessage(result.response, false);
      
      // If there's a code suggestion, offer to apply it
      if (result.code_suggestion) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'chat-message ai';
        
        // Create buttons
        const applyButton = document.createElement('button');
        applyButton.textContent = '✅ Insert Code';
        applyButton.className = 'btn btn-small btn-primary';
        applyButton.style.marginRight = '10px';
        applyButton.onclick = () => {
          // Insert at cursor position or replace selection
          const selection = editor.getSelection();
          const position = editor.getPosition();
          
          if (selection && !selection.isEmpty()) {
            // Replace selected text
            editor.executeEdits('autofill', [
              {
                range: selection,
                text: result.code_suggestion
              }
            ]);
          } else {
            // Insert at current cursor position
            editor.executeEdits('autofill', [
              {
                range: new monaco.Range(position.lineNumber, position.column, position.lineNumber, position.column),
                text: result.code_suggestion
              }
            ]);
          }
          
          messageDiv.remove();
          addChatMessage(`✅ Code inserted successfully!`, false);
          updateStatus('Code autofilled');
        };
        
        const replaceButton = document.createElement('button');
        replaceButton.textContent = '🔄 Replace All';
        replaceButton.className = 'btn btn-small btn-secondary';
        replaceButton.style.marginRight = '10px';
        replaceButton.onclick = () => {
          editor.setValue(result.code_suggestion);
          messageDiv.remove();
          addChatMessage(`✅ Code replaced with generated content!`, false);
          updateStatus('Code replaced');
        };
        
        const previewButton = document.createElement('button');
        previewButton.textContent = '👀 Preview';
        previewButton.className = 'btn btn-small btn-secondary';
        previewButton.onclick = () => {
          const previewDiv = document.createElement('div');
          previewDiv.className = 'code-preview';
          previewDiv.innerHTML = `<pre><code>${escapeHtml(result.code_suggestion)}</code></pre>`;
          messageDiv.appendChild(previewDiv);
          previewButton.remove();
        };
        
        // Add explanation
        const explanationP = document.createElement('p');
        explanationP.innerHTML = '<strong>Generated code is ready!</strong> Choose how to use it:';
        explanationP.style.marginBottom = '10px';
        
        messageDiv.appendChild(explanationP);
        messageDiv.appendChild(applyButton);
        messageDiv.appendChild(replaceButton);
        messageDiv.appendChild(previewButton);
        document.getElementById('chat').appendChild(messageDiv);
      } else {
        addChatMessage("I generated a response but couldn't extract code. Please try a more specific request.", false);
      }
    } else {
      addChatMessage(`Sorry, I couldn't generate code for "${userRequest}". Please try again with a different request.`, false);
    }
    
  } catch (error) {
    addChatMessage(`Error connecting to AI service for code generation. Please check your connection.`, false);
  }
}

// Helper function to escape HTML
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

async function explainCode() {
  await performAIAction('explain');
}

async function debugCode() {
  await performAIAction('debug');
}

async function optimizeCode() {
  await performAIAction('optimize');
}

// Clear functions
function clearEditor() {
  editor.setValue(getDefaultCode(currentLanguage));
  updateStatus('Editor cleared');
}

function clearOutput() {
  document.getElementById('output').textContent = '';
  updateStatus('Output cleared');
}

function clearChat() {
  document.getElementById('chat').innerHTML = '';
  addChatMessage('👋 Hello! I\'m your AI coding assistant powered by the backend server. How can I help you today?', false);
}

function formatCode() {
  editor.getAction('editor.action.formatDocument').run();
  updateStatus('Code formatted');
}

// Check backend connection
async function checkBackendConnection() {
  try {
    const response = await fetch(`${BACKEND_URL}/health`);
    if (response.ok) {
      updateStatus('Backend connected');
      addChatMessage('✅ Connected to AI backend server! All features are available.', false);
    } else {
      throw new Error('Backend unhealthy');
    }
  } catch (error) {
    updateStatus('Backend offline');
    addChatMessage('⚠️ Backend server is not available. Code execution and AI features may not work. Please start the backend server.', false);
  }
}

// Reload API key from environment variables
async function reloadApiKey() {
  const button = document.getElementById('reload-api-key');
  const originalText = button.innerHTML;
  
  try {
    // Show loading state
    button.innerHTML = '<span class="btn-icon">⏳</span>Reloading...';
    button.disabled = true;
    
    const response = await fetch(`${BACKEND_URL}/reload-api-key`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    const result = await response.json();
    
    if (response.ok && result.success) {
      addChatMessage('✅ API key reloaded successfully! AI features should work with updated quota.', false);
      updateStatus('API key reloaded');
    } else {
      throw new Error(result.error || 'Failed to reload API key');
    }
  } catch (error) {
    console.error('Error reloading API key:', error);
    addChatMessage(`❌ Failed to reload API key: ${error.message}`, false);
    updateStatus('API reload failed');
  } finally {
    // Restore button state
    button.innerHTML = originalText;
    button.disabled = false;
  }
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
  // Initialize editor
  initializeEditor();
  
  // Check backend connection
  setTimeout(checkBackendConnection, 2000);
  
  // Language selector
  document.getElementById('language').addEventListener('change', function() {
    changeLanguage(this.value);
  });
  
  // Buttons
  document.getElementById('run-btn').addEventListener('click', runCode);
  document.getElementById('reload-api-key').addEventListener('click', reloadApiKey);
  document.getElementById('clear-editor').addEventListener('click', clearEditor);
  document.getElementById('clear-output').addEventListener('click', clearOutput);
  document.getElementById('clear-chat').addEventListener('click', clearChat);
  document.getElementById('format-code').addEventListener('click', formatCode);
  
  // Chat functionality
  document.getElementById('chat-send').addEventListener('click', sendChatMessage);
  document.getElementById('chat-input').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      sendChatMessage();
    }
  });
  
  // AI actions
  document.getElementById('auto-fill').addEventListener('click', autoFillCode);
  document.getElementById('explain-code').addEventListener('click', explainCode);
  document.getElementById('debug-code').addEventListener('click', debugCode);
  document.getElementById('optimize-code').addEventListener('click', optimizeCode);
  
  // Add welcome message
  setTimeout(() => {
    addChatMessage('👋 Welcome to the AI Virtual Coding Platform with Gemini AI!', false);
    addChatMessage('✨ New Features:', false);
    addChatMessage('🤖 Smart Autofill: Click "Auto-Fill Code" and describe what you want (e.g., "create a calculator", "make a sorting function")', false);
    addChatMessage('🔧 Structured AI: All AI features now provide organized, executable code', false);
    addChatMessage('💡 Self-contained Code: Generated code runs without external files or arguments', false);
    addChatMessage('📝 Try asking: "create a password generator" or "make a file reader with sample data"', false);
  }, 1000);
  
  // Initial status
  updateStatus('Ready');
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
  // Ctrl+Enter to run code
  if (e.ctrlKey && e.key === 'Enter') {
    e.preventDefault();
    runCode();
  }
  
  // Ctrl+Shift+F to format
  if (e.ctrlKey && e.shiftKey && e.key === 'F') {
    e.preventDefault();
    formatCode();
  }
  
  // Ctrl+Shift+C to clear chat
  if (e.ctrlKey && e.shiftKey && e.key === 'C') {
    e.preventDefault();
    clearChat();
  }
});

// Add some CSS for new elements
const style = document.createElement('style');
style.textContent = `
.thinking {
  opacity: 0.7;
  font-style: italic;
}

.code-preview {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 15px;
  border-radius: 8px;
  margin-top: 15px;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 0.85rem;
  overflow-x: auto;
  border: 1px solid #444;
}

.code-preview h4 {
  color: #569cd6;
  margin-top: 0;
  margin-bottom: 10px;
  font-size: 0.9rem;
}

.code-preview pre {
  margin: 0;
  white-space: pre-wrap;
  line-height: 1.4;
}

.action-buttons {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 8px;
  padding: 15px;
  margin-top: 10px;
}

.action-buttons p {
  margin: 0 0 10px 0;
  font-weight: 500;
  color: #4CAF50;
}

.btn.btn-small {
  padding: 8px 16px;
  font-size: 0.85rem;
  border-radius: 6px;
  border: none;
  cursor: pointer;
  font-weight: 500;
  transition: all 0.2s ease;
}

.btn.btn-primary {
  background: #4CAF50;
  color: white;
}

.btn.btn-primary:hover {
  background: #45a049;
  transform: translateY(-1px);
}

.btn.btn-secondary {
  background: #6c757d;
  color: white;
}

.btn.btn-secondary:hover {
  background: #5a6268;
  transform: translateY(-1px);
}

.chat-message.ai {
  animation: fadeInUp 0.3s ease;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(10px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
`;
document.head.appendChild(style);
