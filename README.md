# Swarm Bot

A multi-agent AI application built with Streamlit, ChromaDB, and OpenAI. Swarm Bot coordinates four specialized AI agents — **Planner**, **Researcher**, **Coder**, and **Tester** — in a sequential/parallel pipeline to plan, research, build, and review solutions to a user's goal.

## How It Works

1. You type a goal into the chat bar (optionally attaching a PDF or TXT file for reference).
2. **Planner**, **Researcher**, and **Coder** run concurrently, each tackling their part of the task.
3. **Tester** reviews all three outputs and replies with a single, natural chat response — either a conversational reply (for casual messages) or a summary plus final code (when something was actually built).
4. The Swarm Canvas in the sidebar shows each agent's status in real time — green when idle, orange while working.

## Tech Stack

- **Streamlit** — UI framework
- **ChromaDB** — vector database for storing and retrieving uploaded reference documents
- **OpenAI API** (`gpt-4o` or 5.6, `text-embedding-3-small`) — powers all four agents and embeddings

## Project Structure

```
SwarmFinalProject/
├── App.py                # Main Streamlit app — UI, layout, and orchestration
├── Planner.py             # Planning agent
├── Researcher.py          # Research agent
├── Coder.py                # Coding agent
├── Tester.py                # Review/QA agent
├── Embedding_Helper.py      # OpenAI embedding helper (used by ChromaDB)
├── Chunking.py                # Text chunking utility
├── Doc_Helper.py                # PDF text extraction
├── fonts/                        # Didot font files used in the UI
│   ├── Didot.otf
│   ├── Didot Bold.otf
│   ├── Didot Italic.otf
│   └── Didot Title.otf
├── final_project_db/               # Local ChromaDB persistent storage (auto-created)
└── .env                              # API key (not committed)
```

## Setup

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd SwarmFinalProject
```

### 2. Install dependencies

```bash
pip install streamlit chromadb openai python-dotenv pypdf
```

### 3. Add your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

### 4. Run the app

```bash
streamlit run App.py
```

The app will open at `http://localhost:8501`.

## Notes

- `final_project_db` is created automatically on first run. If you ever change the embedding provider or model, delete this folder and let it regenerate — ChromaDB locks in the embedding function used at creation time and will throw a conflict error otherwise.
- Uploaded PDF/TXT files are chunked and stored in ChromaDB so agents can reference them during a run.
- This project has no error handling by design — it's meant to fail loudly during development rather than silently.

## License

Add your license here.
