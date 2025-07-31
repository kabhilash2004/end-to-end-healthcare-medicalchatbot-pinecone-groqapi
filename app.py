from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

chatModel = ChatOpenAI(
    model="llama3-70b-8192",  # You can also try mixtral-8x7b or gemma-7b
    openai_api_key=GROQ_API_KEY,
    openai_api_base="https://api.groq.com/openai/v1"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

Youtube_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, Youtube_chain)

@app.route("/")
def home():
    """Renders the home page (index.html)."""
    return render_template("index.html")

@app.route('/about')
def about():
    """Renders the about us page (about.html)."""
    return render_template('about.html')


@app.route('/doctors')
def doctors():
    """Renders the doctors page (doctors.html)."""
    return render_template('doctors.html')

@app.route('/contact')
def contact():
    """Renders the contact page (contact.html)."""
    return render_template('contact.html')


@app.route("/chat")
def chat_interface():
    """Renders the chatbot page (chat.html)."""
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat_logic():
    """Handles the user's message and returns the bot's response."""
    msg = request.form["msg"]
    print("User Query:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)