from langchain_community.vectorstores import Chroma
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from rich import print
from langchain_community.llms import ChatGLM
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# 加载文档


load_dotenv()
def load_document(document_path="./b.txt"):
    loader = TextLoader(document_path,encoding='utf-8')
    #    loader = DirectoryLoader(document_path)
    print('loader',loader)
    documents = loader.load()
    print('documents',documents)

    text_spliter = CharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    spliter_docs = text_spliter.split_documents(documents)

    print("文档读取完成~")

    return spliter_docs


def load_embedding_mode():
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cpu"}
    print("embedding加载完成~")

    return HuggingFaceEmbeddings(
        # embedding 模型地址
        model_name="./m3e-base",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def store_chroma(docs, embeddings, persist_directory="VectorStore"):
    # 创建数据库
    db = Chroma.from_documents(docs, embeddings, persist_directory=persist_directory)
    db.persist()
    print("数据库创建完成~")
    return db


model = ChatOpenAI(
    model = 'glm-4',
    openai_api_base = "https://open.bigmodel.cn/api/paas/v4/",
    max_tokens = 500,
    temperature = 0.7
)

embeddings = load_embedding_mode()
# 判断是否已经创建过数据库，如果没有，则创建新的数据库，存在则使用已经存在的
if not os.path.exists('VectorStore'):
    documents = load_document()
    db = store_chroma(documents, embeddings)
else:
    db = Chroma(persist_directory="VectorStore", embedding_function=embeddings)

# 加载模型 ChatGLM
# llm = ChatGLM(
#     endpoint='http://127.0.0.1:8000',
#     max_token=80000,
#     top_p=0.9
# )

retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type='stuff',
    retriever=retriever
)
# 提问文档相关的问题
# The function `run` was deprecated in LangChain 0.1.0 and will be removed in 0.2.0. Use invoke instead.
response = qa.invoke('文章中的故事发生在那个皇帝在位期间？公元多少年？')
print(response)