"""
Test Gemini API connection and functionality.
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

def test_gemini_api():
    """Test if Gemini API key is working."""
    
    print("=" * 60)
    print("TESTING GEMINI API")
    print("=" * 60)
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file")
        return False
    
    print(f"\n✓ API Key found: {api_key[:20]}...")
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        print("✓ API configured successfully")
        
        # Use gemini-2.5-flash (latest available model)
        model_name = 'models/gemini-2.5-flash'
        print(f"\n✓ Using model: {model_name}")
        model = genai.GenerativeModel(model_name)
        
        # Test with a simple prompt
        test_prompt = "Say 'API test successful' in exactly 3 words."
        print(f"\n📤 Testing with prompt: '{test_prompt}'")
        
        response = model.generate_content(test_prompt)
        print(f"📥 Response: {response.text}")
        
        # Test with complaint summarization
        print("\n" + "-" * 60)
        print("Testing complaint summarization:")
        print("-" * 60)
        
        test_complaint = """I opened a credit card account in 2019 and have been making 
        regular payments. However, I noticed unauthorized charges on my account last month. 
        I contacted customer service but they were unhelpful and refused to investigate."""
        
        summary_prompt = f"""Analyze this customer complaint and provide:
1. A 1-2 line summary
2. Category (choose one: Billing, App Issue, Delivery, Support, Other)
3. Urgency (choose one: Low, Medium, High)

Complaint:
{test_complaint}

Format your response exactly as:
Summary: [your summary]
Category: [category]
Urgency: [urgency level]"""
        
        print(f"\n📤 Test complaint: {test_complaint[:100]}...")
        response = model.generate_content(summary_prompt)
        print(f"\n📥 Summary response:\n{response.text}")
        
        print("\n" + "=" * 60)
        print("✅ GEMINI API TEST SUCCESSFUL")
        print("=" * 60)
        print("\nThe API key is working correctly!")
        print("You can now proceed with generating summaries for all complaints.")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ GEMINI API TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        print("\nPossible issues:")
        print("  - Invalid API key")
        print("  - API quota exceeded")
        print("  - Network connection issues")
        print("  - Model not available")
        return False


if __name__ == "__main__":
    success = test_gemini_api()
    
    if success:
        print("\n" + "=" * 60)
        print("NEXT STEP:")
        print("=" * 60)
        print("\nRun the summarization script:")
        print("  python generate_summaries.py")
    else:
        print("\n" + "=" * 60)
        print("TROUBLESHOOTING:")
        print("=" * 60)
        print("\n1. Verify your API key at: https://makersuite.google.com/app/apikey")
        print("2. Check if the API key has proper permissions")
        print("3. Ensure you have API quota available")
