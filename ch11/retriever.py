from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# 임베딩 모델 선언하기
from langchain_openai import OpenAIEmbeddings
embedding = OpenAIEmbeddings(model='text-embedding-3-large',  api_key=OPENAI_API_KEY)

# 언어 모델 불러오기
from langchain_ollama import ChatOllama
llm = ChatOllama(model="deepseek-r1:1.5b")

# Load Chroma store
from langchain_chroma import Chroma
print("Loading existing Chroma store")
persist_directory = '/Users/euni/SrcRepo/hsmu/llm-programming/chroma_store'

vectorstore = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embedding
)

# Create retriever
retriever = vectorstore.as_retriever(k=3)

# Create document chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser # 문자열 출력 파서를 불러옵니다.

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "사용자의 질문에 대해 아래 context에 기반하여 답변하라.:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(llm, question_answering_prompt) | StrOutputParser()

# query augmentation chain
query_augmentation_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="messages"), # 기존 대화 내용
        (
            "system",
            "기존의 대화 내용을 활용하여 사용자의 아래 질문의 의도를 파악하여 명료한 한 문장의 질문으로 변환하라. 대명사나 이, 저, 그와 같은 표현을 명확한 명사로 표현하라. :\n\n{query}",
        ),
    ]
)

query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()


from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    # 1. 질문 입력 및 관련 문서 검색
    query = "여기에 문서 내용과 관련된 질문을 적어보세요"
    docs = retriever.invoke(query)
    print(f"검색된 문서 개수: {len(docs)}")

    answer = document_chain.invoke({
        "context": docs,
        "messages": [HumanMessage(content=query)]
    })
    print("[답변 결과]")
    print(answer)
