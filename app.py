import streamlit as st
import tempfile
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# 페이지 설정
st.set_page_config(
    page_title="📈 주식 포트폴리오 AI 분석",
    page_icon="📊",
    layout="wide"
)

# 커스텀 CSS
st.markdown("""
<style>
    .stChat message {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3a5f, #2d5a87);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown("""
<div class="main-header">
    <h1>📈 주식 포트폴리오 AI 분석 챗봇</h1>
    <p>PDF 문서 기반 RAG 시스템 | Powered by Gemini 2.5 Flash</p>
</div>
""", unsafe_allow_html=True)

# API 키 설정
try:
    GOOGLE_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("⚠️ API 키가 설정되지 않았습니다. Streamlit Secrets에 'GEMINI_API_KEY'를 추가해주세요.")
    st.stop()

# 시스템 프롬프트
SYSTEM_PROMPT = """당신은 주식 포트폴리오 분석 전문가 AI 어시스턴트입니다.

[역할]
- 업로드된 PDF 문서의 내용을 바탕으로 주식 및 투자 관련 질문에 답변합니다.
- 시황 분석, 포트폴리오 구성, 리스크 관리에 대한 조언을 제공합니다.

[답변 원칙]
1. 반드시 제공된 문서(Context)의 내용을 기반으로 답변하세요.
2. 문서에 없는 정보에 대해서는 "제공된 문서에서 해당 정보를 찾을 수 없습니다."라고 명확히 말하세요.
3. 투자 조언 시 항상 "투자의 책임은 본인에게 있습니다"라는 면책 조항을 포함하세요.
4. 답변은 구체적이고 체계적으로 작성하세요.

[Context]
{context}

[대화 기록]
{chat_history}

[사용자 질문]
{question}

[답변]
"""

@st.cache_resource
def get_llm():
    """Gemini 2.5 Flash 모델 초기화"""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3,
        convert_system_message_to_human=True
    )

@st.cache_resource
def get_embeddings():
    """임베딩 모델 초기화"""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )

def process_pdf(pdf_file):
    """PDF 파일 처리 및 벡터 스토어 생성"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    try:
        # PDF 로드
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = text_splitter.split_documents(documents)

        # 벡터 스토어 생성
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)

        return vectorstore, len(documents), len(splits)

    finally:
        os.unlink(tmp_path)

def create_chain(vectorstore):
    """대화형 검색 체인 생성"""
    llm = get_llm()

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False
    )

    return chain

# 사이드바
with st.sidebar:
    st.header("📂 문서 업로드")

    uploaded_file = st.file_uploader(
        "PDF 파일을 업로드하세요",
        type=["pdf"],
        help="주식 리포트, 포트폴리오 문서 등"
    )

    # 기본 test.pdf 사용 옵션
    use_default = st.checkbox("기본 test.pdf 사용", value=False)

    if use_default and os.path.exists("test.pdf"):
        with open("test.pdf", "rb") as f:
            uploaded_file = f
            st.success("✅ test.pdf 로드됨")

    st.divider()

    # 문서 처리
    if uploaded_file is not None:
        if st.button("🔄 문서 처리 시작", type="primary", use_container_width=True):
            with st.spinner("📄 PDF 분석 중..."):
                try:
                    vectorstore, num_pages, num_chunks = process_pdf(uploaded_file)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.chain = create_chain(vectorstore)
                    st.session_state.doc_processed = True

                    st.success(f"""
                    ✅ 문서 처리 완료!
                    - 총 페이지: {num_pages}
                    - 분할 청크: {num_chunks}
                    """)
                except Exception as e:
                    st.error(f"❌ 오류 발생: {str(e)}")

    st.divider()

    # 정보 패널
    st.markdown("""
    <div class="sidebar-info">
    <h4>💡 사용 방법</h4>
    <ol>
        <li>PDF 파일 업로드</li>
        <li>'문서 처리 시작' 클릭</li>
        <li>채팅으로 질문하기</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-info">
    <h4>📊 질문 예시</h4>
    <ul>
        <li>현재 시장 상황은 어떤가요?</li>
        <li>추천 포트폴리오 구성은?</li>
        <li>리스크 관리 방법은?</li>
        <li>섹터별 전망을 알려주세요</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 메인 채팅 영역
col1, col2 = st.columns([3, 1])

with col1:
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "doc_processed" not in st.session_state:
        st.session_state.doc_processed = False

    # 채팅 컨테이너
    chat_container = st.container()

    with chat_container:
        # 이전 메시지 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])

        # 문서 미처리 시 안내
        if not st.session_state.doc_processed:
            st.info("👈 먼저 사이드바에서 PDF 문서를 업로드하고 처리해주세요.")

    # 사용자 입력
    if prompt := st.chat_input("주식 포트폴리오에 대해 질문하세요...", disabled=not st.session_state.doc_processed):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑‍💼"):
            st.markdown(prompt)

        # AI 응답 생성
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("분석 중..."):
                try:
                    chain = st.session_state.chain
                    response = chain({"question": prompt})
                    answer = response["answer"]

                    # 소스 문서 정보
                    sources = response.get("source_documents", [])

                    st.markdown(answer)

                    # 참조 문서 표시
                    if sources:
                        with st.expander("📚 참조 문서"):
                            for i, doc in enumerate(sources[:3]):
                                st.markdown(f"**[{i+1}]** {doc.page_content[:200]}...")

                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"죄송합니다. 오류가 발생했습니다: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

with col2:
    st.markdown("### 📈 빠른 분석")

    if st.session_state.doc_processed:
        quick_questions = [
            "📊 전체 요약",
            "💰 투자 추천",
            "⚠️ 리스크 분석",
            "🔮 시장 전망"
        ]

        for q in quick_questions:
            if st.button(q, use_container_width=True):
                # 버튼 클릭 시 해당 질문으로 채팅
                question_map = {
                    "📊 전체 요약": "문서의 전체 내용을 요약해주세요.",
                    "💰 투자 추천": "현재 추천하는 투자 전략과 포트폴리오 구성은 무엇인가요?",
                    "⚠️ 리스크 분석": "현재 시장의 주요 리스크 요인을 분석해주세요.",
                    "🔮 시장 전망": "향후 시장 전망에 대해 알려주세요."
                }
                st.session_state.quick_question = question_map[q]
                st.rerun()
    else:
        st.caption("문서 처리 후 이용 가능합니다.")

# 빠른 질문 처리
if "quick_question" in st.session_state:
    prompt = st.session_state.quick_question
    del st.session_state.quick_question

    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        chain = st.session_state.chain
        response = chain({"question": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"오류: {str(e)}"})

    st.rerun()

# 푸터
st.divider()
st.caption("⚠️ 본 서비스는 참고용이며, 실제 투자 결정은 전문가와 상담 후 본인의 책임하에 이루어져야 합니다.")
