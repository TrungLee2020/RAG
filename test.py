"""
Script để kiểm tra chức năng retrieval từ vector database

Cách sử dụng:
    python test.py --index-path ./vector_db/index.faiss --documents-path ./vector_db/documents.pkl

    hoặc chạy chế độ tương tác:
    python test.py --interactive
"""

import os
import sys
import argparse
import logging
import json
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from dotenv import load_dotenv

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Tải environment variables
load_dotenv()

# Đường dẫn mặc định
DEFAULT_INDEX_PATH = os.getenv("INDEX_PATH", "vector_db/hnsw.index")
DEFAULT_DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "vector_db/documents.pkl")
DEFAULT_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Alibaba-NLP/gte-multilingual-base")


from retriever import VectorDBRetriever

retriever = VectorDBRetriever(DEFAULT_INDEX_PATH, DEFAULT_DOCUMENTS_PATH)

def display_result(result: Dict[str, Any], show_full_text: bool = False):
    """
    Hiển thị kết quả tìm kiếm dưới dạng dễ đọc

    Args:
        result: Kết quả tìm kiếm
        show_full_text: Có hiển thị toàn bộ văn bản hay không
    """
    print("\n" + "="*80)
    print(f"📄 ID: {result['id']}")
    print(f"🔍 Độ tương đồng: {result['similarity']:.4f}")

    # Hiển thị metadata
    if result['metadata']:
        print("\n📋 Metadata:")
        for key, value in result['metadata'].items():
            if key != 'document_id' and key != 'chunk_id' and key != 'chunk_index' and key != 'total_chunks':
                if isinstance(value, str) and len(value) > 100:
                    print(f"  - {key}: {value[:100]}...")
                else:
                    print(f"  - {key}: {value}")

    # Hiển thị nội dung
    print("\n📝 Nội dung:")
    if show_full_text:
        print(result['document'])
    else:
        # Hiển thị tóm tắt (tối đa 5 dòng đầu tiên)
        lines = result['document'].split('\n')
        print('\n'.join(lines[:5]))
        if len(lines) > 5:
            print("...")
            print(f"(Còn {len(lines) - 5} dòng nữa...)")

    print("="*80)


def interactive_mode(retriever: VectorDBRetriever):
    """
    Chế độ tương tác cho phép người dùng nhập câu truy vấn và xem kết quả

    Args:
        retriever: Đối tượng VectorDBRetriever
    """
    # import readline  # Hỗ trợ chỉnh sửa dòng lệnh

    print("\n🔍 Chế độ tìm kiếm tương tác")
    print("📋 Gõ 'exit', 'quit', hoặc Ctrl+C để thoát")
    print("📋 Gõ 'settings' để thay đổi cài đặt tìm kiếm")

    top_k = 5
    threshold = 0.0
    show_full_text = False

    while True:
        try:
            query = input("\n🔍 Nhập câu truy vấn: ").strip()

            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Tạm biệt!")
                break

            elif query.lower() == 'settings':
                print("\n⚙️ Cài đặt hiện tại:")
                print(f"  - Top K: {top_k}")
                print(f"  - Threshold: {threshold}")
                print(f"  - Hiển thị toàn bộ văn bản: {show_full_text}")

                try:
                    new_top_k = input("  Nhập số lượng kết quả mới (Enter để giữ nguyên): ").strip()
                    if new_top_k:
                        top_k = int(new_top_k)

                    new_threshold = input("  Nhập ngưỡng tương đồng mới (0-1, Enter để giữ nguyên): ").strip()
                    if new_threshold:
                        threshold = float(new_threshold)

                    new_show_full = input("  Hiển thị toàn bộ văn bản? (y/n, Enter để giữ nguyên): ").strip().lower()
                    if new_show_full in ['y', 'yes']:
                        show_full_text = True
                    elif new_show_full in ['n', 'no']:
                        show_full_text = False

                    print("\n⚙️ Cài đặt đã được cập nhật:")
                    print(f"  - Top K: {top_k}")
                    print(f"  - Threshold: {threshold}")
                    print(f"  - Hiển thị toàn bộ văn bản: {show_full_text}")

                except (ValueError, TypeError) as e:
                    print(f"❌ Lỗi khi cập nhật cài đặt: {e}")

                continue

            elif not query:
                continue

            # Thực hiện tìm kiếm
            print(f"\n🔍 Đang tìm kiếm cho: '{query}'")
            results = retriever.retrieve(query, top_k=top_k, threshold=threshold)

            if not results:
                print("❌ Không tìm thấy kết quả nào.")
                continue

            print(f"✅ Tìm thấy {len(results)} kết quả:")

            for i, result in enumerate(results):
                print(f"\n📄 Kết quả #{i+1} (Điểm: {result['similarity']:.4f})")

                # Hiển thị một phần nhỏ của metadata
                if result['metadata']:
                    meta_preview = {}
                    for key, value in result['metadata'].items():
                        if key in ['document_id', 'title', 'source', 'filename']:
                            if isinstance(value, str):
                                meta_preview[key] = value[:50] + "..." if len(value) > 50 else value
                            else:
                                meta_preview[key] = value
                    print(f"📋 Metadata: {meta_preview}")

                # Hiển thị một phần nhỏ của nội dung
                content_preview = result['document'].replace('\n', ' ')[:100]
                print(f"📝 Nội dung: {content_preview}...")

            # Cho phép xem chi tiết một kết quả
            while True:
                choice = input("\n👉 Nhập số kết quả để xem chi tiết (hoặc Enter để quay lại): ").strip()

                if not choice:
                    break

                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(results):
                        display_result(results[choice_idx], show_full_text)
                    else:
                        print(f"❌ Số không hợp lệ. Vui lòng nhập số từ 1 đến {len(results)}.")
                except ValueError:
                    print("❌ Vui lòng nhập một số hợp lệ.")

        except KeyboardInterrupt:
            print("\n👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"❌ Lỗi: {e}")


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra chức năng retrieval từ vector database")

    parser.add_argument("--index-path", type=str, default=DEFAULT_INDEX_PATH,
                        help=f"Đường dẫn tới FAISS index (mặc định: {DEFAULT_INDEX_PATH})")

    parser.add_argument("--documents-path", type=str, default=DEFAULT_DOCUMENTS_PATH,
                        help=f"Đường dẫn tới file documents (mặc định: {DEFAULT_DOCUMENTS_PATH})")

    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME,
                        help=f"Tên model SentenceTransformer (mặc định: {DEFAULT_MODEL_NAME})")

    parser.add_argument("--query", "-q", type=str,
                        help="Câu truy vấn để tìm kiếm")

    parser.add_argument("--top-k", "-k", type=int, default=5,
                        help="Số lượng kết quả trả về (mặc định: 5)")

    parser.add_argument("--threshold", "-t", type=float, default=0.0,
                        help="Ngưỡng điểm tương đồng tối thiểu (0-1) (mặc định: 0.0)")

    parser.add_argument("--full-text", "-f", action="store_true",
                        help="Hiển thị toàn bộ văn bản kết quả")

    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Chạy chế độ tương tác")

    args = parser.parse_args()

    try:
        import time
        start_time = time.time()
        # Tạo retriever
        retriever = VectorDBRetriever(
            index_path=args.index_path,
            documents_path=args.documents_path,
            model_name=args.model_name
        )

        # Chạy chế độ tương tác nếu được yêu cầu
        if args.interactive:
            interactive_mode(retriever)
            return

        # Kiểm tra query
        if not args.query:
            parser.print_help()
            print("\nLỗi: Phải cung cấp một câu truy vấn (--query) hoặc chạy chế độ tương tác (--interactive)")
            return

        # Thực hiện tìm kiếm
        logger.info(f"Đang tìm kiếm cho query: {args.query}")
        results = retriever.retrieve(args.query, top_k=args.top_k, threshold=args.threshold)

        if not results:
            print("Không tìm thấy kết quả nào!")
            return

        # Hiển thị kết quả
        print(f"\nTìm thấy {len(results)} kết quả:")
        print("Total time: ", time.time() - start_time)
        for i, result in enumerate(results):
            print(f"\n--- Kết quả #{i+1} ---")
            display_result(result, args.full_text)

    except Exception as e:
        logger.error(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()