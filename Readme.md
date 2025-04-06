# 🧠 Revision AI

**AI-powered Notion-integrated revision assistant for students, professionals, and lifelong learners.**

![Streamlit Preview](https://img.shields.io/badge/Built%20with-Streamlit-blue?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-green?logo=langchain)
![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-orange?logo=qdrant)

## 🚀 What is Revision AI?

Revision AI is a smart revision tool that connects to your Notion workspace, pulls your notes, and turns them into **interactive quiz questions** or answers your queries using **RAG (Retrieval-Augmented Generation)** with LLMs.

It's like your personalized tutor—trained on your own notes!

---

## ✨ Features

- 🔗 **Seamless Notion Integration**: Sync and load content from any Notion page.
- 🧠 **Context-Aware Q&A**: Ask natural language questions about any Notion note.
- 📝 **AI-Generated Quizzes**: Generate MCQs, short answers, or code-related questions from your notes.
- 📅 **Smart Revision Reminders**: Get reminders based on spaced repetition scheduling.
- 🧭 **Topic-Based Filtering**: Organize and revise notes by custom topics or tags.
- 📲 **Mobile & Desktop Friendly UI**: Clean chat-style interface using Streamlit.
- ⚡ **Powered by LangChain + Groq + Qdrant**: Efficient, fast, and scalable RAG pipeline.

---

## 🛠️ Tech Stack

| Component   | Tool/Library            |
| ----------- | ----------------------- |
| UI          | Streamlit               |
| LLM         | Groq (via LangChain)    |
| Vector DB   | Qdrant                  |
| Embeddings  | Sentence Transformers   |
| Notion Sync | Notion SDK              |
| Scheduler   | Custom JSON-based logic |

---

## 📦 Installation

1. **Clone the repo**

```bash
git clone https://github.com/your-username/revision-ai.git
cd revision-ai
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up your `.env` file**

Create a `.env` file in the root directory with the following variables:

```env
NOTION_TOKEN=your_secret_notion_token
GROQ_API_KEY=your_groq_api_key
QDRANT_HOST=https://your-qdrant-endpoint
QDRANT_API_KEY=your_qdrant_api_key
```

4. **Run the app**

```bash
streamlit run app.py
```

---

## 🌍 Deployment

Revision AI is ready for **Streamlit Cloud deployment**.

1. Push your code to GitHub
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud) and deploy
3. Add your secrets using **Streamlit's Secrets Manager** instead of uploading `.env`

```toml
# .streamlit/secrets.toml
NOTION_TOKEN = "..."
GROQ_API_KEY = "..."
QDRANT_HOST = "..."
QDRANT_API_KEY = "..."
```

---

## 🧪 Example Prompts

- `quiz` — Generates a set of questions from the current Notion page
- `What are the main differences between supervised and unsupervised learning?`
- `Explain this Python function using simple words.`

---

## 🙌 Contributing

We welcome issues, ideas, and PRs. Feel free to open a discussion or contribute!

---

## 📄 License

MIT License

---

## 🌟 Show some love

If you like this project, give it a ⭐️ on GitHub and share it with friends!

---
