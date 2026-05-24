
#  Medical Chatbot

A RAG-based medical chatbot powered by the **Gale Encyclopedia of Medicine**, built with LangChain, Groq (LLaMA 3.3 70B), and Pinecone. Supports conversation memory and personal medical report uploads.

---

##  Features

- **Medical Q&A** — answers questions using the Gale Encyclopedia of Medicine as a knowledge base
- **Conversation Memory** — remembers previous messages within a session, so follow-up questions work naturally
- **Patient Report Upload** — upload your own PDF (blood test, medical report) and ask questions about it; answers are drawn from both your report and the encyclopedia combined
- **Fast inference** — powered by Groq's free-tier LLaMA 3.3 70B model

---

##  Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq — LLaMA 3.3 70B (`langchain-groq`) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store (encyclopedia) | Pinecone (cosine, 384-dim, AWS us-east-1) |
| Vector Store (uploaded report) | FAISS (in-memory) |
| Retrieval | LangChain `ConversationalRetrievalChain` + `MergerRetriever` |
| Memory | `ConversationBufferMemory` |
| Backend | Flask |
| Frontend | HTML + Bootstrap 4 + jQuery |
| Python | 3.11.9 |

---

## 📁 Project Structure

```
Medical-Chatbot/
├── src/
│   ├── helper.py        # PDF loading, chunking, embeddings
│   └── prompt.py        # System prompt
├── templates/
│   └── chat.html        # Chat UI
├── static/
│   └── style.css        # Styling
├── data/                # Gale Encyclopedia PDF
├── research/
│   └── trials.ipynb     # Experimentation notebook
├── app.py               # Flask app — routes, chains, memory
├── store_index.py       # One-time script to index encyclopedia into Pinecone
├── requirements.txt
├── setup.py
└── .env                 # API keys (not committed)
```

---

##  Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/srithichatterjee/Medical-Chatbot.git
cd Medical-Chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv medibot
# Windows (Git Bash)
source medibot/Scripts/activate
# macOS / Linux
source medibot/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root directory:

```
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

- Get a free Pinecone API key at [pinecone.io](https://www.pinecone.io)
- Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Index the encyclopedia into Pinecone (one-time only)

Place the Gale Encyclopedia PDF inside the `data/` folder, then run:

```bash
python store_index.py
```

This creates a Pinecone index named `medical-chatbot` with 384-dimensional cosine embeddings.

### 6. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:8080** in your browser.

---

## 🚀 Usage

### Basic medical Q&A
Just type any medical question in the chat box. The bot answers using the encyclopedia.

### Follow-up questions
Memory is active by default — you can ask follow-ups without repeating context:
```
You:  What is diabetes?
Bot:  Diabetes mellitus is a disorder of carbohydrate metabolism...

You:  What are its symptoms?
Bot:  Symptoms include fatigue, hyperglycemia... (knows "its" = diabetes)
```

### Uploading a personal report
Click **📎 Upload report (PDF)** in the chat footer and select your PDF (blood test, lab report, etc.). Once uploaded, the bot answers using both your report and the encyclopedia together:
```
You:  My report says my HbA1c is 7.2, is that bad?
Bot:  Your HbA1c of 7.2% is above the normal threshold of 6.5%,
      which indicates diabetes. This means your blood sugar has
      been elevated over the past 3 months...
```

---

## 📦 Requirements

```
langchain==0.3.26
langchain-core==0.3.86
langchain-community==0.3.26
langchain-huggingface==0.1.2
langchain-pinecone==0.2.8
langchain-groq==0.2.3
langchain-text-splitters==0.3.11
flask==3.1.1
sentence-transformers==4.1.0
pypdf==5.6.1
python-dotenv==1.1.0
pinecone
faiss-cpu
-e .
```

---

## ⚠️ Known Limitations

- **Memory is session-scoped** — restarting the server clears conversation history
- **Report is session-scoped** — the uploaded PDF is held in memory and lost on restart; re-upload after a restart
- **Single user** — memory and uploaded report are shared across all browser tabs in the same server process; not suitable for multi-user deployment without session isolation
- Running in Flask debug mode may trigger auto-reload and reset the in-memory report index

---


## 🙏 Acknowledgements

- [Gale Encyclopedia of Medicine](https://www.gale.com) — knowledge base
- [LangChain](https://www.langchain.com) — RAG framework
- [Groq](https://groq.com) — fast LLM inference
- [Pinecone](https://www.pinecone.io) — vector database
- [HuggingFace](https://huggingface.co) — embeddings model