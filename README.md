# end_to_end-medical_chatbot
This is a full-stack, RAG-based web application that provides intelligent, context-aware answers to medical questions. It leverages a modern AI stack to deliver a fast, accurate, and conversational user experience.

#Key Features
Retrieval-Augmented Generation (RAG): Delivers accurate answers grounded in a knowledge base of medical documents, minimizing hallucinations and providing trustworthy information.
Real-Time Responses: Integrates with the Groq API to leverage the Llama 3 LLM for ultra-low-latency conversations.
Vector Search: Utilizes Pinecone as a vector database for efficient and scalable semantic search across medical documents.
Full-Stack Application: Complete with a responsive frontend built with HTML/CSS/JS and a robust Python/Flask backend

Tech Stack
Backend: Python, Flask, Gunicorn
AI Orchestration: LangChain
LLM: Groq API (Llama 3)
Embeddings: Hugging Face Inference API
Vector Database: Pinecone
Frontend: HTML, CSS, JavaScript, Bootstrap, Tailwind CSS

Running Locally

STEPS:
Clone the repository

STEP 01- Create a conda environment after opening the repository
conda create -n medibot python=3.10 -y
conda activate medibot

STEP 02- install the requirements
pip install -r requirements.txt

Create a .env file in the root directory and add your Pinecone & Groq api credentials as follows:
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
 
# run the following command to store embeddings to Pinecone
python store_index.py

**Planned Deployment:** AWS

# Finally run the following command
python app.py

Now,
open up localhost:
