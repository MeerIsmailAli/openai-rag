from langchain_community import document_loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import hub
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chat_models import ChatOpenAI
import os




def format_docs(docs):
    return "\n\n".join(doc.page_content.strip() for doc in docs)    

client = MilvusClient(uri="http://localhost:19530")
print("connection to milvus:success")
loader = PyPDFDirectoryLoader("./docs/")

data = loader.load()


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
all_splits = text_splitter.split_documents(data)

vectorstore = Milvus.from_documents(documents=all_splits, embedding=embeddings, collection_name="demo") #test
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
prompt = hub.pull("rlm/rag-prompt")

#llm = OpenAI(temperature=0.7, model_name="gpt-4")
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", max_tokens = 200)

question=input("input the question to feed into chatgpt")
similar_docs = retriever.get_relevant_documents(question)

formatted_docs = format_docs(similar_docs)
print("Top similar documents:\n", formatted_docs)


# #below is for non-openAI
# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

#output_string = ""
# for chunk in rag_chain.stream(question):
#     output_string += chunk
# print(output_string)
# Execute the RAG chain without streaming

rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)


# output_string = rag_chain(question)
# print(output_string)

ans=rag_chain({"question":question,
               "chat_history":""})
print(ans)