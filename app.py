from flask import Flask, render_template, request
from src.helper import download_embeddings, load_pdf_files, text_split
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.retrievers import MergerRetriever
from dotenv import load_dotenv
from src.prompt import *
import os
import tempfile

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

pinecone_retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chatModel = ChatGroq(model="llama-3.3-70b-versatile")

# Global: holds the active retriever (pinecone only, or merged)
active_retriever = pinecone_retriever

def make_chain(retriever):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=chatModel,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        combine_docs_chain_kwargs={
            "prompt": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template("{question}")
            ])
        }
    )

rag_chain = make_chain(pinecone_retriever)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/upload", methods=["POST"])
def upload():
    global rag_chain, active_retriever

    file = request.files.get("report")
    if not file or not file.filename.endswith(".pdf"):
        return {"status": "error", "message": "Please upload a valid PDF."}, 400

    # Save to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    # Load + chunk the uploaded PDF
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    chunks = text_split(docs)

    # Build an in-memory FAISS index from the report
    report_vectorstore = FAISS.from_documents(chunks, embeddings)
    report_retriever = report_vectorstore.as_retriever(search_kwargs={"k": 3})

    # Merge: report + pinecone
    active_retriever = MergerRetriever(retrievers=[report_retriever, pinecone_retriever])

    # Rebuild chain with fresh memory so context is clean for the new report
    rag_chain = make_chain(active_retriever)

    os.unlink(tmp_path)
    return {"status": "ok", "message": f"'{file.filename}' uploaded. You can now ask questions about it."}


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print(msg)
    response = rag_chain.invoke({"question": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)