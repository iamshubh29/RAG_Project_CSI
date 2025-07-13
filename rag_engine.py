import os
import requests
from typing import List, Dict
import json
from dotenv import load_dotenv

load_dotenv()  # Load API keys from .env

class RAGEngine:
    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        if not self.api_key:
            raise ValueError("❌ OPENROUTER_API_KEY not found in .env")

    def generate_response(self, query: str, relevant_documents: List[Dict], model: str = "anthropic/claude-3-haiku") -> str:
        context = self._prepare_context(relevant_documents)
        prompt = self._create_prompt(query, context)

        if "No relevant documents found." in context:
            return "⚠️ Sorry, I couldn't find any relevant content in your uploaded documents to answer this question."

        try:
            response = self._call_openrouter_api(prompt, model)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _prepare_context(self, documents: List[Dict]) -> str:
        if not documents:
            return "No relevant documents found."
        return "\n\n".join(
            f"Document {i+1} (from {doc['metadata'].get('filename')}):\n{doc['content']}"
            for i, doc in enumerate(documents)
        )

    def _create_prompt(self, query: str, context: str) -> str:
        return f"""You are an intelligent assistant that answers questions based on provided documents.

Context:
{context}

Question: {query}
Answer:"""

    def _call_openrouter_api(self, prompt: str, model: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app-url.com",
            "X-Title": "RAG Chatbot"
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7
        }

        response = requests.post(self.base_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        raise Exception(f"API call failed: {response.status_code} - {response.text}")
