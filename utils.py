from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

def qa_agent(openai_api_key, memory, uploaded_file, question):
    model = ChatOpenAI(model="gpt-4o", openai_api_key=openai_api_key)

    #用户上传的文件直接储存在内存，没有一个路径可以给加载器，解决办法：把读入到内存的文件，写入到本地文件，再把这个本地文件的路径传给加载器

    file_content = uploaded_file.read()   #返回bytes 文件内容的二进制数据
    temp_file_path = "temp.pdf"
    with open(temp_file_path, "wb") as temp_file:  #写入是二进制，所以模式是wb
        temp_file.write(file_content)
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n",",",".","!","?","，","。","！","？",""]
    )
    texts = text_splitter.split_documents(docs)
    embeddings_model = OpenAIEmbeddings()
    db = FAISS.from_documents(texts,embeddings_model)
    retriever = db.as_retriever()
    qa =ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory
    )

    response = qa.invoke({"chat_history": memory, "question": question})
    return response

