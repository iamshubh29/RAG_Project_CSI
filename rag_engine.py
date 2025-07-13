import os
import requests
from typing import List, Dict
import json

class RAGEngine:
    def __init__(self):
        self.api_key = os.getenv('OPENROUTER_API_KEY', 'your-api-key-here')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate_response(self, query: str, relevant_documents: List[Dict], model: str = "anthropic/claude-3-haiku") -> str:
        """Generate a response using RAG with OpenRouter API."""

        context = self._prepare_context(relevant_documents)
        prompt = self._create_prompt(query, context)

        if "No relevant documents found." in context:
            return "⚠️ Sorry, I couldn't find any relevant content in your uploaded documents to answer this question. Try uploading a more detailed file or rephrasing your question."

        try:
            response = self._call_openrouter_api(prompt, model)
            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _prepare_context(self, documents: List[Dict]) -> str:
        if not documents:
            return "No relevant documents found."

        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get('content', '')
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            context_parts.append(f"Document {i} (from {filename}):\n{content}")

        return "\n\n".join(context_parts)

    def _create_prompt(self, query: str, context: str) -> str:
        return f"""You are an intelligent assistant that answers questions based on provided documents. 
Use the following context to answer the user's question. If the answer cannot be found in the context, 
say so and provide general guidance if possible.

Context:
{context}

Question: {query}

Please provide a comprehensive and accurate answer based on the context provided. If you reference 
specific information, mention which document it came from."""

    def _call_openrouter_api(self, prompt: str, model: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://your-app-url.com",
            "X-Title": "RAG Chatbot"
        }

        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1000,
            "temperature": 0.7
        }

        response = requests.post(self.base_url, headers=headers, json=data)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")
