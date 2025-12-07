
# 📚 DocuChat — AI-Powered Conversational PDF Assistant

DocuChat is an intelligent RAG-based PDF assistant that allows users to **upload documents and interact with them like a chat**. Instead of scrolling through hundreds of pages, you can ask:

> *"Explain chapter 2"*,
> *"What are the key formulas?"*,
> *"Summarize the conclusions"*,
> *"What is the PDF talking about?"*

…and DocuChat finds the answer instantly.

---

## 🚀 Why This Project?

As engineering students, we've all experienced the last-minute panic:

📄 **200-page PDFs**
⏳ **1 night before the exam**
🥲 **Zero motivation to scroll through everything**

This frustration became the inspiration for **DocuChat** — a tool that lets anyone **study smarter, not harder**.

---

## 🧠 Features

* 📁 **Upload Multiple PDFs**
* 🤖 **Conversational Retrieval-Augmented Generation (RAG)**
* 🧵 **Persistent Session-based chat history**
* ⚡ **Powered by Groq LLM + HuggingFace embeddings**
* 🔍 **Semantic Search over long documents**
* 🖥️ **Modern Gradio Interface**
* 🧠 **Understands context, reformulates questions, and retrieves relevant answers**

---

## 🛠️ Tech Stack

| Technology                     | Purpose                              |
| ------------------------------ | ------------------------------------ |
| **Python**                     | Core language                        |
| **Gradio**                     | Frontend UI                          |
| **LangChain**                  | RAG pipeline + chat memory           |
| **ChromaDB**                   | Vector storage                       |
| **HuggingFace Embeddings**     | Text embeddings (`all-MiniLM-L6-v2`) |
| **Groq API (Gemma / Mixtral)** | LLM response generation              |
| **PyPDFLoader**                | Document parsing                     |

---

## ⚙️ Installation

Clone the repo:

```bash
git clone https://github.com/yourusername/DocuChat.git
cd DocuChat
```

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file:

```
HUGGING_FACE_API_KEY=your_hf_api_key
```

---

## ▶️ Run the App

```bash
gradio app2.py
```

Then open:

👉 `http://127.0.0.1:7860/` in your browser.

---

## 🧪 How to Use

1. Enter your **Groq API key**
2. Upload one or more **PDFs**
3. Click **Build Knowledge Base**
4. Start chatting — ask questions like:

```
What is the conclusion?
Explain page 23.
Give me a summary of chapter 5.
```

---

## 📌 Future Enhancements

* 🔊 Voice-based querying
* 📱 Mobile UI
* 🔗 Support more file formats (DOCX, PPT, Research papers)
* 🧑‍🎓 Personalized study summaries + flashcards

---

## 🤝 Contribution

Pull requests are welcome!
If you'd like to propose major changes, please open an issue first.

---

## ❤️ Credits

Built with love, curiosity, and desperation one night before exams 😄

---

### ⭐ If DocuChat helped you study better — don’t forget to star the repo!

---


