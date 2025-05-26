"""
Phân tách văn bản theo header với xử lý bảng
"""
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import re
from typing import List, Tuple, Optional, Dict, Any
from dotenv import load_dotenv
import logging
import os

# Tải environment variables
load_dotenv()

TOKENIZE_MODEL = os.getenv("TOKENIZE_MODEL")
# Thiết lập logger
logger = logging.getLogger(__name__)


def clean_line_breaks(content):
    """
    Làm sạch các dấu ngắt dòng trong nội dung
    """
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    return '\n'.join(line for line in content.split('\n') if line.strip())


def is_table_header(line):
    """
    Kiểm tra xem một dòng có phải là header của bảng không
    """
    return line.strip().startswith('|') and line.strip().endswith('|') and '-' in line


def is_table_row(line):
    """
    Kiểm tra xem một dòng có phải là hàng của bảng không
    """
    return line.strip().startswith('|') and line.strip().endswith('|')


def extract_tables(content):
    """
    Trích xuất các bảng từ nội dung markdown
    """
    lines = content.splitlines()
    in_table = False
    table_start = 0
    tables = []
    new_lines = []

    for i, line in enumerate(lines):
        # Kiểm tra bắt đầu bảng
        if not in_table and is_table_row(line) and i + 1 < len(lines) and is_table_header(lines[i + 1]):
            in_table = True
            table_start = i
            new_lines.append(f"[TABLE_{len(tables)}]")
        # Kiểm tra kết thúc bảng
        elif in_table and not is_table_row(line):
            in_table = False
            table_content = '\n'.join(lines[table_start:i])
            tables.append(table_content)

        # Thêm các dòng không phải bảng vào nội dung mới
        if not in_table and not (i > 0 and is_table_row(lines[i - 1]) and not is_table_header(line)):
            new_lines.append(line)

    # Xử lý trường hợp bảng kết thúc ở dòng cuối cùng
    if in_table:
        table_content = '\n'.join(lines[table_start:])
        tables.append(table_content)

    return '\n'.join(new_lines), tables


def split_large_table(table_content, max_rows=20):
    """
    Chia các bảng lớn thành các phần nhỏ hơn
    """
    lines = table_content.strip().split('\n')
    header_end = next((i for i, line in enumerate(lines) if is_table_header(line)), -1)

    if header_end == -1:
        return [table_content]

    header = '\n'.join(lines[:header_end + 1])
    body_lines = lines[header_end + 1:]

    return [f"{header}\n" + '\n'.join(body_lines[i:i + max_rows])
            for i in range(0, len(body_lines), max_rows)]


def reinsert_tables(content, tables):
    """
    Đặt lại các bảng vào nội dung
    """
    for i, table in enumerate(tables):
        content = content.replace(f"[TABLE_{i}]", table)
    return content


def preprocess_markdown_with_placeholder(text, headers, placeholder="UKN"):
    """
    Tiền xử lý văn bản markdown để đảm bảo các header trống có nội dung giữ chỗ
    """
    lines = text.splitlines()
    processed_lines = []

    for i, line in enumerate(lines):
        processed_lines.append(line)

        # Kiểm tra xem dòng có phải là header không
        if any(line.startswith(symbol) for symbol, _ in headers):
            # Nếu là dòng cuối hoặc dòng tiếp theo là header cùng cấp/cao hơn
            if i + 1 >= len(lines) or any(
                    lines[i + 1].startswith(symbol) and len(lines[i + 1].split()[0]) <= len(line.split()[0])
                    for symbol, _ in headers
            ):
                # Thêm nội dung giữ chỗ
                processed_lines.append(placeholder)

    return "\n".join(processed_lines)


# def custom_split(content, tokenizer):
#     """
#     Tách văn bản thành các đoạn nhỏ dựa trên cấu trúc header và độ dài
#     """
#     # Làm sạch các dấu ngắt dòng
#     content = clean_line_breaks(content)
#
#     # Trích xuất bảng và thay thế bằng placeholder
#     content_without_tables, tables = extract_tables(content)
#
#     # Xử lý bảng
#     processed_tables = []
#     for table in tables:
#         table_chunks = split_large_table(table)
#         processed_tables.extend(table_chunks)
#
#     # Định nghĩa recursive character splitter cho các đoạn không có header
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1024,
#         chunk_overlap=256,
#         length_function=lambda x: len(tokenizer.encode(x)),
#         separators=["\n\n", "\n", " ", ""]
#     )
#
#     # Định nghĩa header để tách (h1-h6)
#     headers_to_split_on = [(f"{'#' * i} ", f"h{i}") for i in range(1, 7)]
#     markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
#
#     # Tiền xử lý nội dung với placeholder cho header trống
#     preprocessed_content = preprocess_markdown_with_placeholder(content_without_tables, headers_to_split_on)
#
#     splits = []
#
#     # Trích xuất header của văn bản (tối đa 3 dòng trước dấu '#' đầu tiên)
#     lines = preprocessed_content.split('\n')
#     doc_header_end = next((i for i, line in enumerate(lines) if line.startswith('#')), len(lines))
#     doc_header = '\n'.join(lines[max(0, doc_header_end - 3):doc_header_end]).strip()
#
#     try:
#         # Thử tách theo header markdown trước
#         markdown_splits = markdown_splitter.split_text(preprocessed_content)
#
#         for split in markdown_splits:
#             # Kết hợp metadata header với nội dung
#             header = "\n".join([f"{k} {v}" for k, v in split.metadata.items() if v])
#             combined_text = f"{header}\n\n{split.page_content.strip()}"
#
#             # Nếu split chỉ chứa placeholder, thêm nó chỉ với header
#             if split.page_content.strip() == "UKN":
#                 splits.append(f"{doc_header}\n\n{header}")
#             else:
#                 # Thay thế lại bảng nếu cần
#                 combined_text_with_tables = reinsert_tables(combined_text, processed_tables)
#
#                 # Tách tiếp các phần lớn bằng RecursiveCharacterTextSplitter
#                 text_splits = text_splitter.split_text(combined_text_with_tables)
#                 splits.extend([f"{doc_header}\n\n{s}" if not s.startswith(doc_header) else s for s in text_splits])
#     except Exception as e:
#         # Nếu không tìm thấy header, sử dụng trực tiếp RecursiveCharacterTextSplitter
#         # Đặt lại bảng trước
#         content_with_tables = reinsert_tables(preprocessed_content, processed_tables)
#         text_splits = text_splitter.split_text(content_with_tables)
#         splits = [f"{doc_header}\n\n{s}" if not s.startswith(doc_header) else s for s in text_splits]
#
#     # Xử lý sau khi tách
#     splits = [s for s in splits if s.strip()]  # Loại bỏ các phần trống
#     splits = [re.sub(r'^\n+', '', s) for s in splits]  # Loại bỏ dòng trống ở đầu
#
#     # Làm sạch header và mẫu cụ thể
#     splits = ['\n'.join(
#         s.split('\n')[:3] +
#         [line for line in s.split('\n')[3:] if line.startswith('#') or not line.strip().startswith('Số:')]
#     ) for s in splits]
#
#     # Loại bỏ placeholder khỏi kết quả cuối cùng
#     splits = [s.replace("\nUKN", "") for s in splits]
#
#     return splits
def custom_split(content: str, tokenizer) -> List[str]:
    """
    Tách văn bản thành các đoạn nhỏ dựa trên cấu trúc header và độ dài
    """
    content = clean_line_breaks(content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
        length_function=lambda x: len(tokenizer.encode(x)),
        separators=["\n\n", "\n", " ", ""]
    )

    headers_to_split_on = [(f"{'#' * i}", f"{'#' * i}") for i in range(1, 7)]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    preprocessed_content = preprocess_markdown_with_placeholder(content, headers_to_split_on)

    # Lấy dòng "Số: ..." làm tiêu đề chính
    lines = preprocessed_content.split('\n')
    doc_header = ''
    for i in range(min(5, len(lines))):
        if re.match(r'^Số:\s*\d+\/[A-Z\-]+', lines[i]):
            doc_header = lines[i]
            break

    splits = []

    try:
        markdown_chunks = markdown_splitter.split_text(preprocessed_content)

        for section in markdown_chunks:
            section_header = "\n".join([f"{k} {v}".strip() for k, v in section.metadata.items() if v])
            section_text = section.page_content.strip()

            if section_text == "UKN":
                # Trường hợp không có nội dung rõ ràng
                full_text = f"{doc_header}\n\n{section_header}"
                splits.append(full_text.strip())
            else:
                # Split nội dung dài thành nhiều đoạn ngắn hơn
                text_chunks = text_splitter.split_text(section_text)
                for chunk in text_chunks:
                    full_text = f"{doc_header}\n\n{section_header}\n\n{chunk}"
                    splits.append(full_text.strip())

    except Exception as e:
        logger.error(f"Markdown splitting failed: {e}")
        # fallback nếu markdown split lỗi
        text_chunks = text_splitter.split_text(preprocessed_content)
        splits = [f"{doc_header}\n\n{chunk.strip()}" for chunk in text_chunks]

    # Loại bỏ placeholder dư thừa
    splits = [s.replace('\nUKN', '').strip() for s in splits if s.strip()]

    return splits

class DocumentChunk:
    """
    Đại diện cho một đoạn văn bản đã được chia nhỏ từ tài liệu gốc
    """

    def __init__(self,
                 text: str,
                 document_id: Optional[str] = None,
                 chunk_id: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Khởi tạo một chunk văn bản

        Args:
            text: Nội dung văn bản của chunk
            document_id: ID của tài liệu gốc
            chunk_id: ID riêng của chunk
            metadata: Thông tin bổ sung về chunk
        """
        self.text = text
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi chunk thành dictionary

        Returns:
            Dict: Dữ liệu của chunk dưới dạng dictionary
        """
        return {
            "text": self.text,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata
        }


if __name__ == "__main__":
    # Example markdown content
    path = "../data_sample/QLCL_txt/596-QD BĐVN Ban hành Quy định công tác kiểm tra kiểm soát chất lượng tại các đơn vị 2023.txt"
    with open(path, encoding="utf-8") as f:
        sample_markdown = f.read()
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

    # Split the document
    chunks = custom_split(sample_markdown, tokenizer)

    # Print results
    print(f"Split document into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)