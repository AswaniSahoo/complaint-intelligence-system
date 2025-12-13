import os
import time
from typing import List, Dict
import google.generativeai as genai
from groq import Groq
from openai import OpenAI


class LLMSummarizer:
    """Summarize and categorize complaints using various LLM APIs."""
    
    def __init__(self, provider='ollama', api_key=None):
        """
        Initialize LLM client.
        
        Args:
            provider: 'gemini', 'groq', or 'ollama'
            api_key: API key (or reads from environment)
        """
        self.provider = provider.lower()
        
        if self.provider == 'gemini':
            api_key = api_key or os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('models/gemini-2.5-flash')
            print("Initialized Gemini API")
        
        elif self.provider == 'groq':
            api_key = api_key or os.getenv('GROQ_API_KEY')
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            self.client = Groq(api_key=api_key)
            self.model_name = 'llama-3.1-8b-instant'
            print("Initialized Groq API")
        
        elif self.provider == 'ollama':
            api_key = api_key or os.getenv('OLLAMA_API_KEY')
            if not api_key:
                raise ValueError("OLLAMA_API_KEY not found in environment")
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.together.xyz/v1"
            )
            self.model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
            print("Initialized Ollama/Together API")
        
        else:
            raise ValueError("Provider must be 'gemini', 'groq', or 'ollama'")
    
    def create_prompt(self, complaint_text):
        """Create prompt for summarization and categorization."""
        prompt = f"""Analyze this customer complaint and provide:
1. A 1-2 line summary
2. Category (choose one: Billing, App Issue, Delivery, Support, Other)
3. Urgency (choose one: Low, Medium, High)

Complaint:
{complaint_text}

Format your response exactly as:
Summary: [your summary]
Category: [category]
Urgency: [urgency level]"""
        return prompt
    
    def parse_response(self, response_text):
        """Parse LLM response into structured data."""
        lines = response_text.strip().split('\n')
        result = {
            'summary': '',
            'category': 'Other',
            'urgency': 'Medium'
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('Summary:'):
                result['summary'] = line.replace('Summary:', '').strip()
            elif line.startswith('Category:'):
                result['category'] = line.replace('Category:', '').strip()
            elif line.startswith('Urgency:'):
                result['urgency'] = line.replace('Urgency:', '').strip()
        
        return result
    
    def process_complaint_gemini(self, complaint_text):
        """Process complaint using Gemini."""
        prompt = self.create_prompt(complaint_text)
        
        try:
            response = self.model.generate_content(prompt)
            parsed = self.parse_response(response.text)
            return parsed
        except Exception as e:
            print(f"Error with Gemini: {e}")
            return {'summary': complaint_text[:100], 'category': 'Other', 'urgency': 'Medium'}
    
    def process_complaint_groq(self, complaint_text):
        """Process complaint using Groq."""
        prompt = self.create_prompt(complaint_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            parsed = self.parse_response(response.choices[0].message.content)
            return parsed
        except Exception as e:
            print(f"Error with Groq: {e}")
            return {'summary': complaint_text[:100], 'category': 'Other', 'urgency': 'Medium'}
    
    def process_complaint_ollama(self, complaint_text):
        """Process complaint using Ollama/Together API."""
        prompt = self.create_prompt(complaint_text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            parsed = self.parse_response(response.choices[0].message.content)
            return parsed
        except Exception as e:
            print(f"Error with Ollama: {e}")
            return {'summary': complaint_text[:100], 'category': 'Other', 'urgency': 'Medium'}
    
    def process_complaint(self, complaint_text):
        """Process a single complaint."""
        if self.provider == 'gemini':
            return self.process_complaint_gemini(complaint_text)
        elif self.provider == 'groq':
            return self.process_complaint_groq(complaint_text)
        else:
            return self.process_complaint_ollama(complaint_text)
    
    def process_batch(self, complaints, batch_size=10, delay=1.0):
        """
        Process multiple complaints with rate limiting.
        
        Args:
            complaints: List of complaint texts
            batch_size: Number to process before delay
            delay: Seconds to wait between batches
        
        Returns:
            List of dicts with summary, category, urgency
        """
        results = []
        total = len(complaints)
        
        print(f"Processing {total} complaints using {self.provider}...")
        
        for i, complaint in enumerate(complaints):
            result = self.process_complaint(complaint)
            results.append(result)
            
            if (i + 1) % batch_size == 0:
                print(f"  Processed {i+1}/{total}")
                time.sleep(delay)
        
        print(f"Completed processing {total} complaints")
        return results


def summarize_complaints(df, text_column='complaint_text', provider='gemini', batch_size=10):
    """
    Add LLM-generated summaries, categories, and urgency to dataframe.
    
    Args:
        df: DataFrame with complaints
        text_column: Column containing complaint text
        provider: 'gemini' or 'groq'
        batch_size: Batch size for processing
    
    Returns:
        DataFrame with added columns
    """
    summarizer = LLMSummarizer(provider=provider)
    
    # Process complaints
    results = summarizer.process_batch(
        df[text_column].tolist(),
        batch_size=batch_size
    )
    
    # Add results to dataframe
    df['llm_summary'] = [r['summary'] for r in results]
    df['llm_category'] = [r['category'] for r in results]
    df['llm_urgency'] = [r['urgency'] for r in results]
    
    return df


if __name__ == "__main__":
    import pandas as pd
    
    # Example usage with a small sample
    df = pd.read_csv("../data/processed/processed_complaints.csv")
    df_sample = df.head(20)
    
    # Try with Gemini
    df_sample = summarize_complaints(df_sample, provider='gemini', batch_size=5)
    
    print("\nSample results:")
    print(df_sample[['complaint_text', 'llm_summary', 'llm_category', 'llm_urgency']].head())
