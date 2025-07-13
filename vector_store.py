import os
from dotenv import load_dotenv
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

# Load environment variables
load_dotenv()

class VectorStore:
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_ANON_KEY')
        self.table_name = "documents"

        if not self.supabase_url or not self.supabase_key:
            print("⚠️ Supabase credentials not found in .env")
            self.supabase: Optional[Client] = None
        else:
            try:
                self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
                print("✅ Supabase connected")
            except Exception as e:
                print(f"❌ Supabase connection failed: {str(e)}")
                self.supabase: Optional[Client] = None

        # ✅ Use 384-dim local model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        self._initialize_database()

    def _initialize_database(self):
        if not self.supabase:
            return
        try:
            self.supabase.table(self.table_name).select("*").limit(1).execute()
        except Exception:
            print(f"""
⚠️ Table '{self.table_name}' might not exist. Please run this SQL in Supabase:

CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
""")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate local embedding (384-dim)."""
        return self.embedding_model.encode(text).tolist()

    def add_documents(self, documents: List[Dict]):
        if not self.supabase:
            print("⚠️ Supabase not configured.")
            return

        for doc in documents:
            try:
                embedding = self._generate_embedding(doc['content'])
                self.supabase.table(self.table_name).insert({
                    'content': doc['content'],
                    'embedding': embedding,
                    'metadata': doc['metadata']
                }).execute()
                print(f"✅ Added: {doc['metadata'].get('filename')}")
                print(f"Storing chunk: {doc['content'][:100]}...")  # Shows first 100 chars
            except Exception as e:
                print(f"❌ Error adding doc: {str(e)}")

    def similarity_search(self, query: str, k: int = 3) -> List[Dict]:
        if not self.supabase:
            return [{
                'content': f"Sample result for: {query}",
                'metadata': {'filename': 'demo.txt'}
            }]
        try:
            query_embedding = self._generate_embedding(query)
            result = self.supabase.rpc('match_documents', {
                'query_embedding': query_embedding,
                'match_threshold': 0.3,
                'match_count': k
            }).execute()
            print(f"Similarity search result: {result.data if result and hasattr(result, 'data') else result}")
            return result.data if result.data else []
        except Exception as e:
            print(f"❌ Similarity search error: {str(e)}")
            return []

    def get_document_count(self) -> int:
        if not self.supabase:
            return 0
        try:
            result = self.supabase.table(self.table_name).select("id", count="exact").execute()
            return result.count if result.count else 0
        except Exception as e:
            print(f"❌ Count error: {str(e)}")
            return 0

    def clear_documents(self):
        if not self.supabase:
            return
        try:
            self.supabase.table(self.table_name).delete().neq('id', 0).execute()
            print("🗑️ Cleared all documents.")
        except Exception as e:
            print(f"❌ Clear failed: {str(e)}")

