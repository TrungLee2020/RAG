import os
import logging
import pickle
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Default paths
DEFAULT_INDEX_PATH = os.getenv("INDEX_PATH", "vector_db/hnsw.index")
DEFAULT_DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "vector_db/documents.pkl")
DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SearchResult:
    """Search result from the retriever"""

    def __init__(self, id: int, document: str, similarity: float, metadata: Dict[str, Any] = None):
        """
        Initialize a search result

        Args:
            id: Document ID
            document: Document content
            similarity: Similarity score (0-1)
            metadata: Additional metadata for the document
        """
        self.id = id
        self.document = document
        self.similarity = similarity
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert search result to a dictionary

        Returns:
            Dict: Result as a dictionary
        """
        return {
            "id": self.id,
            "document": self.document,
            "similarity": self.similarity,
            "metadata": self.metadata
        }


class VectorDBRetriever:
    """
    Retriever for documents based on SentenceTransformer and FAISS
    """

    def __init__(
            self,
            index_path: str = DEFAULT_INDEX_PATH,
            documents_path: str = DEFAULT_DOCUMENTS_PATH,
            model_name: str = DEFAULT_MODEL_NAME
    ):
        """
        Initialize the retriever

        Args:
            index_path: Path to FAISS index file
            documents_path: Path to documents pickle file
            model_name: SentenceTransformer model name
        """
        self.index_path = index_path
        self.documents_path = documents_path
        self.model_name = model_name

        # Check if files exist
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Index file not found: {index_path}")

        if not os.path.exists(documents_path):
            raise FileNotFoundError(f"Documents file not found: {documents_path}")

        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

        # Load index and documents
        self.index = self._load_index(index_path)
        self.documents, self.metadata = self._load_documents(documents_path)

        logger.info(f"Successfully loaded. Document count: {len(self.documents)}")

    def _load_index(self, index_path: str) -> faiss.Index:
        """
        Load FAISS index from file

        Args:
            index_path: Path to index file

        Returns:
            faiss.Index: Loaded index
        """
        try:
            logger.info(f"Loading FAISS index from: {index_path}")
            index = faiss.read_index(index_path)

            # Configure efSearch for HNSW index to improve search accuracy
            if hasattr(index, 'hnsw'):
                index.hnsw.efSearch = 128
                logger.info(f"Set efSearch = 128 for HNSW index")
            return index
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise

    def _load_documents(self, documents_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Load documents and metadata from pickle file

        Args:
            documents_path: Path to documents file

        Returns:
            Tuple[List[str], List[Dict]]: Documents and metadata
        """
        try:
            logger.info(f"Loading documents from: {documents_path}")
            with open(documents_path, 'rb') as f:
                data = pickle.load(f)

            # Support multiple data formats
            if isinstance(data, tuple) and len(data) == 2:
                # If tuple (documents, metadata)
                documents, metadatas = data
                logger.info(f"Loaded data in tuple format")
                return documents, metadatas
            elif isinstance(data, dict):
                # If dict with 'documents' and 'metadata'/'metadatas' keys
                documents_key = 'documents'
                metadata_key = 'metadatas' if 'metadatas' in data else 'metadata'

                if documents_key in data and metadata_key in data:
                    logger.info(f"Loaded data in dictionary format")
                    return data[documents_key], data[metadata_key]
                else:
                    raise ValueError(f"Invalid data format in {documents_path}")
            elif isinstance(data, list):
                # If just a list of documents, create empty metadata
                logger.info(f"Loaded data as list, creating empty metadata")
                return data, [{} for _ in range(len(data))]
            else:
                raise ValueError(f"Unrecognized data format in {documents_path}")
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise

    def retrieve(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve documents most relevant to the query

        Args:
            query: Query string
            top_k: Number of results to return
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List[Dict]: List of search results
        """
        if not query.strip():
            logger.warning("Empty query, cannot search")
            return []

        if self.index.ntotal == 0:
            logger.warning("Empty vector store, cannot search")
            return []

        # Create embedding for query
        query_vector = self.model.encode(query)
        query_vector = np.array([query_vector], dtype=np.float32)

        # Normalize query vector for cosine similarity
        faiss.normalize_L2(query_vector)

        # Search for results
        logger.debug(f"Searching top {top_k} results for query: {query}")
        distances, indices = self.index.search(query_vector, top_k)

        # Process results
        results = []

        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Skip invalid results
                continue

            # Calculate similarity from distance
            # For normalized vectors, inner product equals cosine similarity
            similarity = float(dist)

            # Check similarity threshold
            if similarity < threshold:
                continue

            # Get document content and metadata
            document = self.documents[idx]
            metadata = self.metadata[idx] if idx < len(self.metadata) else {}

            # Create result
            result = SearchResult(
                id=idx,
                document=document,
                similarity=similarity,
                metadata=metadata
            ).to_dict()

            results.append(result)

        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 5, threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
        """
        Retrieve documents for multiple queries at once

        Args:
            queries: List of query strings
            top_k: Number of results to return per query
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List[List[Dict]]: List of results for each query
        """
        if not queries:
            logger.warning("Empty queries list, cannot search")
            return []

        if self.index.ntotal == 0:
            logger.warning("Empty vector store, cannot search")
            return [[] for _ in queries]

        # Create embeddings for all queries
        query_vectors = self.model.encode(queries)
        query_vectors = np.array(query_vectors, dtype=np.float32)

        # Normalize query vectors for cosine similarity
        faiss.normalize_L2(query_vectors)

        # Search for similar documents
        distances, indices = self.index.search(query_vectors, top_k)

        # Process results
        all_results = []

        for query_idx, (query_distances, query_indices) in enumerate(zip(distances, indices)):
            results = []

            for i, (dist, idx) in enumerate(zip(query_distances, query_indices)):
                if idx == -1:  # Skip invalid results
                    continue

                # Calculate similarity from distance
                similarity = float(dist)

                # Check similarity threshold
                if similarity < threshold:
                    continue

                # Get document content and metadata
                document = self.documents[idx]
                metadata = self.metadata[idx] if idx < len(self.metadata) else {}

                # Create result
                result = SearchResult(
                    id=idx,
                    document=document,
                    similarity=similarity,
                    metadata=metadata
                ).to_dict()

                results.append(result)

            all_results.append(results)

        return all_results
