from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List

class VectorStore:
    """Manages vector store operations"""

    def __init__(self):
        """Initialize vector store with OpenAI embeddings"""
        self.embeddings = OpenAIEmbeddings()
        self.retriever = None
        self.vectorstore = None

    def create_vectorstore(self, documents: List[Document]):
        """Creates vector store from documents
        
        Args:
            documents: List of documents to embed
        """

        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever()

    def get_retriever(self):
        """Returns the retriever of this vectorstore
        
        Return:
            Retriever instance
        """

        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query, top_k: int = 4) -> List[Document]:
        """Returns the top 4 documents for the query asked
        
        Args:
            query: User query
            top_k: Number of top k documents to retrieve
        
        Return:
            List of documents
        """
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)