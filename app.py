import os
import tempfile

# LangChain 관련 라이브러리
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts  import ChatPromptTemplate

import streamlit as st

from dotenv import load_dotenv # .env 파일의 환경변수를 자동으로 불러오기 위한 모듈
load_dotenv()  # 실행 시 .env 파일을 찾아 변수들을 환경에 로드

# --------------------------------------------------
# PDF → VectorDB 변환 함수 (캐싱 적용)
# --------------------------------------------------
    
# 캐싱 처리
@st.cache_resource
def build_vector_db(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    os.remove(pdf_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)

    return vectordb

# --------------------------------------------------
#  RAG 기반 답변 생성
# --------------------------------------------------
def get_response(query, vectorstore,chat_history):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 
    system_prompt = (
        "당신은 보험 약관을 고객의 눈높이에서 쉽고 친절하게 설명해주는 'AI 보험 가이드'입니다."
        "아래 제공된 [Context]를 바탕으로 사용자의 질문에 답변해 주세요."
        "[지침 사항]"
        "1. 톤앤매너: 딱딱한 말투 대신, 고객을 대하듯 따뜻하고 부드러운 문체를 사용하세요."
        "2. 사실 기반: 반드시 [Context]에 명시된 내용만으로 답변하세요. 약관에 없는 내용은 절대 추측하지 말고 "
        "'약관 내용에서 찾을 수 없어요'라고 솔직하게 말하세요."
        "3. 출처 명시: 답변의 끝에는 반드시 근거가 되는 '관련 조항(제 몇 조)'이나 '페이지'를 언급해 주세요."
        "4. 표현: '약관에 따르면 ~라고 되어 있어요'라는 객관적인 표현을 사용하세요."
        "5. 대상: 성인 대상이지만 보험에 대해서는 잘 모르는 사람을 대상으로 쉽게 잘 풀어서 설명해 줘야 한다."
        "[Context]: {context}"
    )  
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{question}")
    ])

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    response = rag_chain.invoke(query)
    return response.content


# --------------------------------------------------
# Session State 초기화
# --------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "안녕하세요! 보험 약관 PDF를 업로드해 주시면 분석해 드릴게요 😊"}
    ]

# --------------------------------------------------
#  Streamlit 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="AI 보험 약관 분석기",
    page_icon="📑",
    layout="wide"
)

main_title = """
AI 보험 약관 분석기는
당신이 가입한 보험 약관(PDF)을 업로드해 두고,
궁금한 내용을 질문하면 답변을 도와주는 서비스입니다.

어려운 보험 용어를 고객의 눈높이에 맞추어 설명해 드리며,
보장 내용・면책 사항・보장 한도 등 핵심 정보를 빠르게 확인하실 수 있어요.

업로드만 해 두면, 나머지는 AI가 알아서 처리해 드릴게요 🙂
"""

# --------------------------------------------------
# 메인 화면
# --------------------------------------------------
st.header("🔍 AI 보험 약관 Q&A")
st.info(f"💡{main_title}")
st.subheader("📄 약관 PDF 업로드")

if st.session_state.vectorstore is None:    
    uploaded_file = st.file_uploader(
        "보험약관 PDF를 올려주세요.",
        type=["pdf"],
        key="upload_pdf"    )
    if uploaded_file and st.button("약관 분석 시작하기", key="add_policy"):     
        with st.spinner("약관을 분석하는 중입니다... (최초 1회만 실행)"):
            try:
                st.session_state.vectorstore = build_vector_db(uploaded_file)
                # print(st.session_state.vectorstore)
                policy_name = uploaded_file.name
                st.success(f"'{policy_name}' 약관이 추가되었습니다. 이제 이 약관으로 질문하실 수 있어요.")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
    else:
        st.stop()
else:
    st.info("이미 벡터 DB가 생성되어 있어요. 다시 만들지 않습니다.")

# 이전 대화 출력
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 질문 입력
query = st.chat_input("보험 약관에 대해 무엇이든 물어보세요!")
if query and st.session_state.vectorstore is not None:
    # 사용자 질문 표시
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # AI 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("약관에서 답을 찾는 중입니다..."):
            answer = get_response(query, st.session_state.vectorstore,st.session_state.chat_history)
            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})