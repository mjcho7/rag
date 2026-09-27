"""DATA 폴더의 문서를 근거로만 답하는 간단한 RAG 챗봇입니다."""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# app.py가 있는 프로젝트 최상단과 DATA 폴더를 기준으로 경로를 정합니다.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "DATA"
SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".csv", ".json"}


def load_documents(data_dir: Path) -> list[Document]:
    """DATA 폴더의 지원 파일을 읽고 파일명·페이지 정보를 함께 저장합니다."""
    documents: list[Document] = []

    for file_path in sorted(path for path in data_dir.rglob("*") if path.is_file()):
        suffix = file_path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue

        if suffix == ".pdf":
            # PDF는 페이지마다 Document를 만들어, 답변의 근거 위치를 더 명확히 합니다.
            reader = PdfReader(str(file_path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": file_path.name, "page": page_number},
                        )
                    )
        else:
            # UTF-8이 아닌 텍스트도 최대한 읽되, 읽지 못한 글자는 대체합니다.
            text = file_path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": file_path.name}))

    return documents


@st.cache_resource(show_spinner="DATA 문서를 읽고 검색 인덱스를 만드는 중입니다...")
def build_vector_store() -> InMemoryVectorStore:
    """문서를 적당한 크기로 나눈 후 OpenAI 임베딩으로 메모리 벡터DB를 만듭니다."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA 폴더를 찾을 수 없습니다: {DATA_DIR}")

    documents = load_documents(DATA_DIR)
    if not documents:
        raise ValueError("DATA 폴더에서 읽을 수 있는 문서를 찾지 못했습니다.")

    # 문장 경계를 우선 보존하면서, 긴 문서를 검색하기 좋은 크기로 나눕니다.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(chunks)
    return vector_store


def evidence_sentence(text: str, question: str) -> str:
    """검색된 조각에서 질문과 가장 겹치는 한 문장을 화면에 보여 줍니다."""
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]
    if not sentences:
        return text.strip()[:300]

    keywords = {word for word in re.findall(r"[가-힣A-Za-z0-9]+", question.lower()) if len(word) > 1}
    best = max(
        sentences,
        key=lambda sentence: sum(keyword in sentence.lower() for keyword in keywords),
    )
    return best[:300] + ("..." if len(best) > 300 else "")


def format_context(documents: list[Document]) -> str:
    """LLM이 파일명과 페이지를 함께 보도록 검색 결과를 문자열로 만듭니다."""
    parts = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "알 수 없는 파일")
        page = document.metadata.get("page")
        location = f"{source} {page}쪽" if page else source
        parts.append(f"[근거 {index}: {location}]\n{document.page_content}")
    return "\n\n".join(parts)


def answer_question(question: str, vector_store: InMemoryVectorStore) -> tuple[str, list[Document]]:
    """최신 메시지 API로 검색 결과를 근거로만 답변합니다."""
    documents = vector_store.similarity_search(question, k=4)
    context = format_context(documents)
    system_prompt = """당신은 제공된 문서 근거만으로 답하는 RAG 도우미입니다.
문서에 답이 없거나 근거가 부족하면 반드시 '제공된 문서에서 확인할 수 없습니다.'라고 답하세요.
문서에 없는 사실, 수치, 규정, 절차를 추측하거나 일반 지식으로 보완하지 마세요.
답변은 한국어로 간결하게 작성하세요."""
    user_prompt = f"""질문: {question}

문서 근거:
{context}
"""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = model.invoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])
    return str(response.content), documents


def main() -> None:
    """Streamlit 화면을 구성합니다."""
    load_dotenv(PROJECT_DIR / ".env")

    st.set_page_config(page_title="공무원 여비 RAG 챗봇", page_icon="📚")
    st.title("📚 공무원 여비 RAG 챗봇")
    st.caption("DATA 폴더 문서를 검색해, 확인 가능한 내용만 답변합니다.")

    if not os.getenv("OPENAI_API_KEY"):
        st.error(".env 파일에 OPENAI_API_KEY를 설정한 뒤 다시 실행하세요.")
        st.code("OPENAI_API_KEY=sk-...", language="bash")
        st.stop()

    try:
        vector_store = build_vector_store()
    except Exception as error:
        st.error(f"문서 검색 인덱스를 만들지 못했습니다: {error}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                show_sources(message["sources"], message.get("question", ""))

    question = st.chat_input("문서에 대해 질문하세요")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("문서에서 근거를 찾는 중입니다..."):
            answer, sources = answer_question(question, vector_store)
        st.markdown(answer)
        show_sources(sources, question)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "question": question}
    )


def show_sources(sources: list[Document], question: str) -> None:
    """답변 바로 아래에 중복을 제거한 파일명과 실제 근거 문장을 표시합니다."""
    st.markdown("#### 출처 및 근거 문장")
    seen: set[tuple[str, int | None, str]] = set()
    for document in sources:
        source = str(document.metadata.get("source", "알 수 없는 파일"))
        page = document.metadata.get("page")
        sentence = evidence_sentence(document.page_content, question)
        key = (source, page, sentence)
        if key in seen:
            continue
        seen.add(key)
        page_text = f", {page}쪽" if page else ""
        st.caption(f"**{source}{page_text}** — {sentence}")


if __name__ == "__main__":
    main()
