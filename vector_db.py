import os
import faiss
import pickle
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import re
from text_splitter import DocumentChunk, custom_split

# Tải environment variables
load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# Thiết lập logger
logger = logging.getLogger(__name__)

class FAISSVectorStore:

    """
    Vector store sử dụng FAISS cho việc lưu trữ và tìm kiếm embedding
    """

    def __init__(self,
                 index_path: Optional[str] = None,
                 documents_path: Optional[str] = None,
                 model_name: str = EMBEDDING_MODEL,
                 dimension: int = None,
                 metric_type: str = "cosine",
                 index_type: str = "hnsw"):
        """
        Khởi tạo FAISS vector store

        Args:
            index_path: Đường dẫn đến file index FAISS (nếu đã có)
            documents_path: Đường dẫn đến file lưu documents và metadata
            model_name: Tên model SentenceTransformer
            dimension: Số chiều của vector embedding (nếu None, sẽ tự xác định từ model)
            metric_type: Loại metric sử dụng ("cosine", "l2", "ip")
            index_type: Loại index FAISS ("flat", "hnsw", "ivf")
        """
        self.model_name = model_name
        self.index_path = index_path
        self.documents_path = documents_path
        self.metric_type = metric_type
        self.index_type = index_type

        # Khởi tạo model embedding
        logger.info(f"Đang tải model SentenceTransformer: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

        # Xác định số chiều
        self.dimension = dimension or self.model.get_sentence_embedding_dimension()
        logger.info(f"Số chiều embedding: {self.dimension}")

        # Khởi tạo FAISS index
        if index_path and os.path.exists(index_path) and documents_path and os.path.exists(documents_path):
            # Tải index và documents nếu đã có
            logger.info(f"Đang tải FAISS index từ: {index_path}")
            self.index = self._load_index(index_path)
            logger.info(f"Đang tải documents từ: {documents_path}")
            self.documents, self.metadatas = self._load_documents(documents_path)
        else:
            # Tạo index mới
            logger.info(f"Tạo FAISS index mới với loại: {index_type}")
            self.index = self._create_index()
            self.documents = []
            self.metadatas = []

    def _create_index(self) -> faiss.Index:
        """
        Tạo FAISS index dựa trên loại được chọn

        Returns:
            faiss.Index: FAISS index đã khởi tạo
        """
        # Xác định metric
        # Chuẩn hóa tất cả vectors nên dùng inner product cho cosine similarity
        if self.metric_type == "cosine":
            metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric_type == "l2":
            metric = faiss.METRIC_L2
        elif self.metric_type == "ip":
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            raise ValueError(f"Metric không hợp lệ: {self.metric_type}. Hỗ trợ: cosine, l2, ip")

        # Tạo index dựa trên loại
        if self.index_type.lower() == "flat":
            # Index đơn giản, chính xác nhưng chậm với dữ liệu lớn
            index = faiss.IndexFlat(self.dimension, metric)

        elif self.index_type.lower() == "hnsw":
            # HNSW index, nhanh hơn và vẫn khá chính xác
            M = 32  # Số kết nối tối đa trên mỗi layer

            index = faiss.IndexHNSWFlat(self.dimension, M, metric)
            # Cài đặt tham số
            index.hnsw.efConstruction = 200  # Chất lượng xây dựng cao hơn (mặc định: 40)
            index.hnsw.efSearch = 128  # Chất lượng tìm kiếm cao hơn (mặc định: 16)

        elif self.index_type.lower() == "ivf":
            # IVF index, nhanh hơn nhưng có thể kém chính xác hơn
            nlist = 100  # Số lượng clusters

            # Cần một index trung gian để clustering
            quantizer = faiss.IndexFlat(self.dimension, metric)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric)
            # IVF cần training

        else:
            raise ValueError(f"Loại index không hợp lệ: {self.index_type}. Hỗ trợ: flat, hnsw, ivf")

        return index

    def _load_index(self, index_path: str) -> faiss.Index:
        """
        Tải FAISS index từ file

        Args:
            index_path: Đường dẫn đến file index

        Returns:
            faiss.Index: FAISS index đã tải
        """
        try:
            index = faiss.read_index(index_path)
            # Cài đặt efSearch cho HNSW
            if hasattr(index, 'hnsw'):
                index.hnsw.efSearch = 128
            return index
        except Exception as e:
            logger.error(f"Lỗi khi tải FAISS index: {e}")
            raise

    def _load_documents(self, documents_path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Tải documents và metadata từ file

        Args:
            documents_path: Đường dẫn đến file documents

        Returns:
            Tuple: (documents, metadatas)
        """
        try:
            with open(documents_path, 'rb') as f:
                data = pickle.load(f)

            # Hỗ trợ nhiều định dạng dữ liệu
            if isinstance(data, tuple) and len(data) == 2:
                return data
            elif isinstance(data, dict) and 'documents' in data and 'metadatas' in data:
                return data['documents'], data['metadatas']
            elif isinstance(data, list):
                if all(isinstance(item, str) for item in data):
                    return data, [{} for _ in range(len(data))]
                elif all(isinstance(item, dict) for item in data):
                    docs = [item.get('text', '') for item in data]
                    metas = [item.get('metadata', {}) for item in data]
                    return docs, metas

            raise ValueError(f"Không nhận dạng được định dạng dữ liệu từ {documents_path}")

        except Exception as e:
            logger.error(f"Lỗi khi tải documents: {e}")
            raise

    def save(self, index_path: str = None, documents_path: str = None):
        """
        Lưu index và documents ra file

        Args:
            index_path: Đường dẫn lưu index (nếu None, dùng self.index_path)
            documents_path: Đường dẫn lưu documents (nếu None, dùng self.documents_path)
        """
        index_path = index_path or self.index_path
        documents_path = documents_path or self.documents_path

        if not index_path or not documents_path:
            raise ValueError("Cần cung cấp đường dẫn để lưu index và documents")

        # Tạo thư mục nếu cần
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(documents_path), exist_ok=True)

        # Lưu index
        logger.info(f"Đang lưu FAISS index vào: {index_path}")
        faiss.write_index(self.index, index_path)

        # Lưu documents và metadata
        logger.info(f"Đang lưu documents và metadata vào: {documents_path}")
        with open(documents_path, 'wb') as f:
            pickle.dump((self.documents, self.metadatas), f)

        logger.info("Đã lưu index và documents thành công")

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa vectors cho cosine similarity

        Args:
            vectors: Vectors cần chuẩn hóa

        Returns:
            np.ndarray: Vectors đã chuẩn hóa
        """
        if self.metric_type == "cosine":
            # Chuẩn hóa vectors để dùng với METRIC_INNER_PRODUCT
            faiss.normalize_L2(vectors)
        return vectors

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None, save: bool = False):
        """
        Thêm texts vào vector store

        Args:
            texts: Danh sách các văn bản cần thêm
            metadatas: Metadata tương ứng với mỗi văn bản
            save: Có lưu index và documents sau khi thêm không
        """
        if not texts:
            logger.warning("Không có texts để thêm")
            return

        # Khởi tạo metadata nếu không được cung cấp
        metadatas = metadatas or [{} for _ in range(len(texts))]

        if len(metadatas) != len(texts):
            raise ValueError(f"Số lượng metadata ({len(metadatas)}) phải bằng số lượng texts ({len(texts)})")

        # Tạo embeddings
        logger.info(f"Đang tạo embeddings cho {len(texts)} văn bản")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Luôn chuẩn hóa embeddings cho cosine similarity
        # Với cosine similarity, cần chuẩn hóa cả vector truy vấn và vector trong index
        # Để bảo đảm tương thích với METRIC_INNER_PRODUCT
        logger.info("Chuẩn hóa embeddings cho cosine similarity")
        faiss.normalize_L2(embeddings)

        # Đào tạo IVF index nếu cần
        if self.index_type.lower() == "ivf" and not self.index.is_trained and len(embeddings) > 0:
            logger.info("Đang đào tạo IVF index")
            self.index.train(embeddings)

        # Thêm vectors vào index
        logger.info(f"Đang thêm {len(embeddings)} vectors vào FAISS index")
        self.index.add(embeddings)

        # Cập nhật documents và metadata
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)

        # Lưu nếu cần
        if save:
            self.save()

    def add_documents(self, chunks: List[DocumentChunk], save: bool = False):
        """
        Thêm DocumentChunks vào vector store

        Args:
            chunks: Danh sách các DocumentChunk cần thêm
            save: Có lưu index và documents sau khi thêm không
        """
        if not chunks:
            logger.warning("Không có chunks để thêm")
            return

        # Chuyển đổi chunks thành texts và metadatas
        texts = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                **chunk.metadata
            }
            for chunk in chunks
        ]

        # Thêm vào vector store
        self.add_texts(texts, metadatas, save)

    def add_document_with_chunks(self, document: str, document_id: str = None, metadata: Dict[str, Any] = None,
                                 save: bool = False):
        """
        Thêm một document, tự động chia nhỏ thành chunks và lưu vào vector store

        Args:
            document: Nội dung văn bản cần thêm
            document_id: ID của document (tạo tự động nếu không cung cấp)
            metadata: Metadata của document
            save: Có lưu index và documents sau khi thêm không
        """
        from uuid import uuid4

        # Tạo document_id nếu không được cung cấp
        document_id = document_id or str(uuid4())
        metadata = metadata or {}

        # Chia document thành các chunks
        chunks = custom_split(document, self.model.tokenizer)

        # Tạo DocumentChunk objects
        doc_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}_{i}"
            chunk = DocumentChunk(
                text=chunk_text,
                document_id=document_id,
                chunk_id=chunk_id,
                metadata={
                    **metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            doc_chunks.append(chunk)

        # Thêm chunks vào vector store
        self.add_documents(doc_chunks, save)

        return doc_chunks

    def similarity_search(self,
                          query: str,
                          top_k: int = 5,
                          threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Tìm kiếm văn bản tương tự với query

        Args:
            query: Câu truy vấn
            top_k: Số lượng kết quả trả về
            threshold: Ngưỡng điểm tương đồng tối thiểu (0-1)

        Returns:
            List[Dict]: Danh sách các kết quả tìm kiếm
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store trống, không thể tìm kiếm")
            return []

        # Tạo embedding cho query
        query_vector = self.model.encode(query)
        query_vector = np.array([query_vector], dtype=np.float32)

        # Luôn chuẩn hóa query vector cho cosine similarity
        # Vì index đã được tạo với metric_type là INNER_PRODUCT cho vectors đã chuẩn hóa
        logger.info("Chuẩn hóa query vector cho cosine similarity")
        faiss.normalize_L2(query_vector)

        # Tìm kiếm kết quả
        logger.info(f"Đang tìm kiếm top {top_k} kết quả cho query: {query}")
        distances, indices = self.index.search(query_vector, top_k)

        # Xử lý kết quả
        results = []

        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:  # Bỏ qua kết quả không hợp lệ
                continue

            # Tính điểm tương đồng từ khoảng cách
            # Với vector đã chuẩn hóa và dùng inner product,
            # giá trị cosine similarity nằm trong khoảng [0, 1]
            similarity = float(dist)

            # Kiểm tra ngưỡng tương đồng
            if similarity < threshold:
                continue

            # Lấy nội dung và metadata
            document = self.documents[idx]
            metadata = self.metadatas[idx] if idx < len(self.metadatas) else {}

            # Tạo kết quả
            result = {
                "id": idx,
                "document": document,
                "similarity": similarity,
                "metadata": metadata
            }

            results.append(result)

        return results

    def _distance_to_similarity(self, distance: float) -> float:
        """
        Chuyển đổi khoảng cách thành điểm tương đồng

        Args:
            distance: Khoảng cách từ FAISS

        Returns:
            float: Điểm tương đồng (0-1)
        """
        if self.metric_type == "cosine" or self.metric_type == "ip":
            # Đối với inner product với vectors đã chuẩn hóa, giá trị đã nằm trong khoảng [0, 1]
            # Vì đã sử dụng normalize_L2, giá trị inner product chính là cosine similarity
            return max(0.0, min(1.0, float(distance)))
        else:  # l2
            # Đối với L2, khoảng cách nhỏ hơn = tương đồng cao hơn
            # Chuyển đổi khoảng cách L2 thành điểm tương đồng
            return 1.0 / (1.0 + distance)

    def batch_similarity_search(self,
                                queries: List[str],
                                top_k: int = 5,
                                threshold: float = 0.0) -> List[List[Dict[str, Any]]]:
        """
        Tìm kiếm hàng loạt cho nhiều queries

        Args:
            queries: Danh sách các câu truy vấn
            top_k: Số lượng kết quả trả về cho mỗi query
            threshold: Ngưỡng điểm tương đồng tối thiểu (0-1)

        Returns:
            List[List[Dict]]: Danh sách kết quả cho mỗi query
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store trống, không thể tìm kiếm")
            return [[] for _ in range(len(queries))]

        # Tạo embeddings cho tất cả queries
        query_vectors = self.model.encode(queries, show_progress_bar=True)
        query_vectors = np.array(query_vectors, dtype=np.float32)

        # Luôn chuẩn hóa query vectors cho cosine similarity
        logger.info("Chuẩn hóa query vectors cho cosine similarity")
        faiss.normalize_L2(query_vectors)

        # Tìm kiếm kết quả
        logger.info(f"Đang tìm kiếm top {top_k} kết quả cho {len(queries)} queries")
        distances, indices = self.index.search(query_vectors, top_k)

        # Xử lý kết quả
        all_results = []

        for query_idx, (query_distances, query_indices) in enumerate(zip(distances, indices)):
            results = []

            for i, (dist, idx) in enumerate(zip(query_distances, query_indices)):
                if idx == -1:  # Bỏ qua kết quả không hợp lệ
                    continue

                # Tính điểm tương đồng từ khoảng cách
                similarity = float(dist)

                # Kiểm tra ngưỡng tương đồng
                if similarity < threshold:
                    continue

                # Lấy nội dung và metadata
                document = self.documents[idx]
                metadata = self.metadatas[idx] if idx < len(self.metadatas) else {}

                # Tạo kết quả
                result = {
                    "id": idx,
                    "document": document,
                    "similarity": similarity,
                    "metadata": metadata
                }

                results.append(result)

            all_results.append(results)

        return all_results

    def delete(self, document_ids: List[str] = None, chunk_ids: List[str] = None):
        """
        Xóa documents khỏi vector store dựa trên document_id hoặc chunk_id

        Args:
            document_ids: Danh sách document_ids cần xóa
            chunk_ids: Danh sách chunk_ids cần xóa
        """
        if not document_ids and not chunk_ids:
            logger.warning("Không có document_ids hoặc chunk_ids để xóa")
            return

        # Tìm các chỉ số cần xóa
        indices_to_remove = []

        for i, metadata in enumerate(self.metadatas):
            # Kiểm tra document_id
            if document_ids and metadata.get("document_id") in document_ids:
                indices_to_remove.append(i)

            # Kiểm tra chunk_id
            elif chunk_ids and metadata.get("chunk_id") in chunk_ids:
                indices_to_remove.append(i)

        if not indices_to_remove:
            logger.warning("Không tìm thấy document nào phù hợp để xóa")
            return

        # Xóa theo chỉ số
        self._delete_by_indices(indices_to_remove)

    def _delete_by_indices(self, indices: List[int]):
        """
        Xóa documents theo chỉ số

        Args:
            indices: Danh sách chỉ số cần xóa
        """
        # Sắp xếp chỉ số giảm dần để xóa từ cuối lên đầu
        indices.sort(reverse=True)

        # Không thể xóa trực tiếp từ FAISS index, phải tạo index mới
        # Lưu trữ vectors còn lại
        remaining_documents = []
        remaining_metadatas = []
        embeddings = []

        # Ghi chú các chỉ số cần xóa
        indices_set = set(indices)

        # Lọc documents và tạo embeddings mới
        for i in range(len(self.documents)):
            if i not in indices_set:
                remaining_documents.append(self.documents[i])
                remaining_metadatas.append(self.metadatas[i])
                # Tạo embedding cho văn bản còn lại
                embedding = self.model.encode(self.documents[i])
                embeddings.append(embedding)

        # Tạo index mới
        new_index = self._create_index()

        # Thêm vectors vào index mới
        if embeddings:
            embeddings = np.array(embeddings).astype('float32')

            # Chuẩn hóa vectors nếu cần
            embeddings = self._normalize_vectors(embeddings)

            # Đào tạo IVF index nếu cần
            if self.index_type.lower() == "ivf" and len(embeddings) > 0:
                new_index.train(embeddings)

            # Thêm vectors
            new_index.add(embeddings)

        # Cập nhật index và documents
        self.index = new_index
        self.documents = remaining_documents
        self.metadatas = remaining_metadatas

        logger.info(f"Đã xóa {len(indices)} documents khỏi vector store")

    def clear(self):
        """
        Xóa tất cả documents khỏi vector store
        """
        # Tạo index mới
        self.index = self._create_index()
        self.documents = []
        self.metadatas = []

        logger.info("Đã xóa tất cả documents khỏi vector store")