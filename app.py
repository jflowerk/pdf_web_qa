import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import openai  # openai 모듈을 import합니다.
from openai_api import get_response_from_openai  # 수정된 get_response_from_openai 함수를 사용
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
from langchain_community.callbacks import get_openai_callback

# 환경 변수에서 OpenAI API 키를 로드하고 설정
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def add_newlines_to_response(text):
    """
    마침표 다음에 줄바꿈 추가. 마침표와 다음 문장 사이의 공간을 유지하기 위해 ". " 대신 ".\n"을 사용
    """
    text_with_newlines = text.replace(". ", ".\n\n")
    return text_with_newlines

def main():
    st.set_page_config(page_title="PDF Q&A")
    st.header("PDF Q&A? 💬")
    
    pdf = st.file_uploader("PDF를 업로드 하세요", type="pdf")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # 사용자 질문 입력 및 처리
        st.write("PDF 내용에 관한 질문을 입력해주세요. 예: 연차휴가에 대해 설명해주세요.'")
        user_question = st.text_input("업로드 PDF기반 질문을 작성해주세요:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                # 예시: max_tokens 값을 512로 설정
                response = chain.run(input_documents=docs, question=user_question, max_tokens=512)

                print(cb)  # 콘솔 로깅을 위한 코드

            response_with_newlines = add_newlines_to_response(response)
            
            st.text_area("답변", value=response_with_newlines, height=300)

if __name__ == '__main__':
    main()
