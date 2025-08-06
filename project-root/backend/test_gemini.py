import google.generativeai as genai
import asyncio

# Configure Gemini AI
GEMINI_API_KEY = "AIzaSyCEfBjtgmX3YCBoMUbnIacHXo6BbsIVMX0"
genai.configure(api_key=GEMINI_API_KEY)

async def test_gemini():
    try:
        print("Testing Gemini API...")
        
        # List available models first
        print("Available models:")
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                print(f"- {model.name}")
        
        # Try different model names
        model_names = [
            'gemini-1.5-flash',
            'gemini-1.5-pro', 
            'models/gemini-1.5-flash',
            'models/gemini-1.5-pro'
        ]
        
        for model_name in model_names:
            try:
                print(f"\nTesting model: {model_name}")
                gemini_model = genai.GenerativeModel(model_name)
                
                # Simple test prompt
                prompt = "Hello! Please respond with a short greeting."
                
                response = gemini_model.generate_content(prompt)
                
                if response and response.text:
                    print(f"Success with {model_name}! Response: {response.text[:100]}...")
                    return model_name
                else:
                    print(f"No response from {model_name}")
                    
            except Exception as e:
                print(f"Error with {model_name}: {str(e)}")
                continue
        
        return None
            
    except Exception as e:
        print(f"General error: {str(e)}")
        return None

if __name__ == "__main__":
    working_model = asyncio.run(test_gemini())
    print(f"\nTest result: {'FOUND WORKING MODEL: ' + working_model if working_model else 'NO WORKING MODEL FOUND'}")
