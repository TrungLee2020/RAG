from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
import pandas as pd
import os
from dotenv import load_dotenv

# Load env
load_dotenv()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

class Retriever:
    def __init__(self, index_path=None, documents_path=None, metadata_csv=None,
                 model_name=EMBEDDING_MODEL):
        # Khởi tạo model và index
        self.model = SentenceTransformer(model_name)
        self.dimension = 384

        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = None

        if documents_path and os.path.exists(documents_path):
            with open(documents_path, 'rb') as f:
                self.id_to_document = pickle.load(f)
        else:
            self.id_to_document = {}

        # Tải metadata
        self.metadata = None
        if metadata_csv and os.path.exists(metadata_csv):
            self.metadata = pd.read_csv(metadata_csv)
            self.metadata_dict = {row['document_id']: row.to_dict() for _, row in self.metadata.iterrows()}

        # Mapping giữa FAISS index ID và document_id
        self.index_to_doc_id = {}

    def get_embedding(self, text):
        """Tạo embedding vector cho một đoạn văn bản"""
        return self.model.encode(text, show_progress_bar=False)

    def build_index_with_metadata(self, documents, metadata_df, output_dir):
        """Xây dựng FAISS index với tích hợp metadata"""
        os.makedirs(output_dir, exist_ok=True)

        # Tạo mappings
        doc_id_to_index = {}
        index_to_doc_id = {}
        id_to_document = {}

        # Xử lý từng tài liệu
        embeddings = []
        valid_docs = []
        valid_metadata = []

        for i, (_, meta_row) in enumerate(metadata_df.iterrows()):
            doc_id = meta_row['document_id']
            filepath = os.path.join(output_dir, '..', meta_row['filepath'])

            # Kiểm tra file tồn tại
            if not os.path.exists(filepath):
                print(f"Warning: File not found: {filepath}")
                continue

            # Đọc nội dung tài liệu
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                print(f"Error reading {filepath}: {str(e)}")
                continue

            # Tạo embedding
            embedding = self.get_embedding(content)

            # Lưu mappings
            doc_id_to_index[doc_id] = i
            index_to_doc_id[i] = doc_id
            id_to_document[i] = content

            # Thêm vào danh sách
            embeddings.append(embedding)
            valid_docs.append(content)
            valid_metadata.append(meta_row)

        embeddings = np.array(embeddings, dtype=np.float32)

        # Tạo HNSW index
        M = 32  # Số kết nối lớn nhất mỗi node
        index = faiss.IndexHNSWFlat(self.dimension, M) # indexing với FAISS sử dùng HNSW cho dữ liệu lớn theo số node

        # Thiết lập tham số
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 40

        # Thêm vectors vào index
        index.add(embeddings)

        # Lưu index và mappings
        index_path = os.path.join(output_dir, "hnsw_meta.index")
        faiss.write_index(index, index_path)

        # Lưu mapping id -> document
        documents_path = os.path.join(output_dir, "documents_meta.pkl")
        with open(documents_path, 'wb') as f:
            pickle.dump(id_to_document, f)

        # Lưu mapping index -> doc_id
        mapping_path = os.path.join(output_dir, "index_to_doc_id.pkl")
        with open(mapping_path, 'wb') as f:
            pickle.dump(index_to_doc_id, f)

        # Cập nhật metadata với chỉ các tài liệu hợp lệ
        valid_meta_df = pd.DataFrame(valid_metadata)
        meta_path = os.path.join(output_dir, "metadata_processed.csv")
        valid_meta_df.to_csv(meta_path, index=False)

        # Cập nhật instance variables
        self.index = index
        self.id_to_document = id_to_document
        self.index_to_doc_id = index_to_doc_id
        self.metadata = valid_meta_df
        self.metadata_dict = {row['document_id']: row.to_dict() for _, row in valid_meta_df.iterrows()}

        print(f"Đã xây dựng index với {index.ntotal} vectors và tích hợp metadata")

        return index_path, documents_path, mapping_path, meta_path

    def build_index_from_chunks(chunks, output_dir, model_name=EMBEDDING_MODEL):
        """Build FAISS index from document chunks with metadata tracking"""
        os.makedirs(output_dir, exist_ok=True)

        model = SentenceTransformer(model_name)
        dimension = 384

        # Tạo mappings
        chunk_to_index = {}
        index_to_chunk = {}
        chunk_metadata = {}

        # Xử lý từng chunk
        texts = [chunk.page_content for chunk in chunks]

        # Tạo embeddings (batch processing)
        print(f"Creating embeddings for {len(texts)} chunks...")
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

        # Lưu metadata
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.metadata["chunk_id"]
            chunk_to_index[chunk_id] = i
            index_to_chunk[i] = chunk_id
            chunk_metadata[chunk_id] = chunk.metadata

        # Tạo HNSW index
        print("Building HNSW index...")
        M = 32  # Number of connections per layer
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 40

        # In thông tin HNSW
        print(index.hnsw)

        # Thêm vectors vào index
        index.add(embeddings)

        # Lưu index và mappings
        index_path = os.path.join(output_dir, "chunks_hnsw.index")
        faiss.write_index(index, index_path)

        # Lưu mapping và metadata
        mappings = {
            "chunk_to_index": chunk_to_index,
            "index_to_chunk": index_to_chunk,
            "chunk_metadata": chunk_metadata,
            "texts": texts
        }

        mappings_path = os.path.join(output_dir, "chunks_mappings.pkl")
        with open(mappings_path, 'wb') as f:
            pickle.dump(mappings, f)

        return index_path, mappings_path

    def retrieve_with_citations(self, query, top_k=5, threshold=0.0):
        """Truy xuất tài liệu với thông tin trích dẫn"""
        if self.index is None:
            raise ValueError("Index has not been initialized")

        # Tạo embedding cho query
        query_embedding = self.get_embedding(query)
        query_embedding = np.array([query_embedding], dtype=np.float32)

        # Tìm kiếm trong index
        distances, indices = self.index.search(query_embedding, top_k)

        # Xử lý kết quả với metadata
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:
                # Tính điểm tương đồng
                similarity = 1 / (1 + float(dist))

                # Lấy thông tin tài liệu
                document = self.id_to_document[int(idx)]

                # Lấy metadata nếu có
                citation_info = {}
                if self.metadata is not None and idx in self.index_to_doc_id:
                    doc_id = self.index_to_doc_id[idx]
                    if doc_id in self.metadata_dict:
                        meta = self.metadata_dict[doc_id]
                        citation_info = {
                            'document_id': doc_id,
                            'title': meta.get('title', ''),
                            'author': meta.get('author', ''),
                            'source': meta.get('source', ''),
                            'date': meta.get('date', ''),
                            'filename': meta.get('filename', ''),
                            'category': meta.get('category', '')
                        }

                if similarity >= threshold:
                    results.append({
                        "id": int(idx),
                        "document": document,
                        "similarity": similarity,
                        "metadata": citation_info
                    })

        return results