## RAG Q&A Conversation With PDF Including Chat History
import os
import gradio as gr
from gtts import gTTS

import streamlit as st  # <- REMOVE THIS IF YOU DON’T NEED IT ANYWHERE ELSE

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from huggingface_hub import HfApi

# ---------------- HF SECRETS SETUP ----------------
# On Hugging Face Spaces, add these in Settings ➜ Repository secrets:
#   HUGGING_FACE_API_KEY
#   GROQ_API_KEY
HF_TOKEN = os.environ.get("HUGGING_FACE_API_KEY")
GROQ_SECRET = os.environ.get("GROQ_API_KEY")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ------------------------------------------------------------------
# GLOBAL STORE (replacement for st.session_state.store)
# ------------------------------------------------------------------
store = {}


def get_session_history(session: str) -> BaseChatMessageHistory:
    """Statefully manage chat history per session (same logic as Streamlit version)."""
    if session not in store:
        store[session] = ChatMessageHistory()
    return store[session]


# ------------------------------------------------------------------
# BUILD RAG PIPELINE ONCE API KEY + PDFS ARE PROVIDED
# ------------------------------------------------------------------
def setup_rag_pipeline(api_key, uploaded_files, session_id):
    """
    Creates the LLM, processes PDFs, builds vectorstore, retriever,
    and conversational_rag_chain. Returns the chain and a status message.
    """

    # Prefer HF secret if available; fall back to textbox value for local dev
    api_key = GROQ_SECRET or api_key

    if not api_key:
        return None, "⚠️ Please enter your Groq API key first."

    if not uploaded_files or len(uploaded_files) == 0:
        return None, "📄 Please upload at least one PDF."

    # LLM
    llm = ChatGroq(groq_api_key=api_key, model_name="groq/compound")

    # Process uploaded PDFs (Gradio-compatible)
    documents = []
    for uploaded_file in uploaded_files:
        # Gradio can return dicts, paths, or file-like objects depending on version
        if isinstance(uploaded_file, dict):
            # Older Gradio: {"name": "/tmp/...", "data": b"...", ...}
            temppdf = uploaded_file.get("name")
        elif isinstance(uploaded_file, str):
            # Sometimes it's already a path string
            temppdf = uploaded_file
        else:
            # File-like object with .name
            temppdf = getattr(uploaded_file, "name", None)

        if not temppdf:
            raise ValueError("Could not determine file path for uploaded PDF.")

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)

    # Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Contextualization prompt (same as your code)
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # QA prompt (same)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain, f"✅ RAG pipeline is ready for session `{session_id}`. Ask your questions below!"


# ------------------------------------------------------------------
# ANSWER QUESTIONS FUNCTION (USED BY GRADIO)
# ------------------------------------------------------------------
def answer_question(user_input, session_id, chain, chat_history_ui):
    """
    Uses the existing conversational_rag_chain to answer questions.
    chat_history_ui is only for Gradio's Chatbot display.
    Real memory is still handled by ChatMessageHistory (same as before).
    """
    # Gradio's Chatbot (in your version) uses "messages" format:
    # [{"role": "user"/"assistant", "content": "..."}]
    if chat_history_ui is None:
        chat_history_ui = []

    if not chain:
        chat_history_ui = chat_history_ui + [
            {"role": "user", "content": user_input},
            {
                "role": "assistant",
                "content": "⚠️ Please enter API key, upload PDFs and click 'Build Knowledge Base' first.",
            },
        ]
        return chat_history_ui

    session_history = get_session_history(session_id)

    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    answer = response["answer"]

    # Append user + assistant messages in dict format
    chat_history_ui = chat_history_ui + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer},
    ]
    return chat_history_ui


# ------------------------------------------------------------------
# GRADIO UI
# ------------------------------------------------------------------
custom_css = """
#app-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #5f27cd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#app-subtitle {
    text-align: center;
    font-size: 1rem;
    opacity: 0.8;
    margin-bottom: 1.5rem;
}
.gradio-container {
    background: radial-gradient(circle at top left, #1e293b 0, #020617 45%, #000000 100%);
    color: #e5e7eb;
}
"""

with gr.Blocks() as demo:
    gr.HTML(f"<style>{custom_css}</style>")
    gr.Markdown("<div id='app-title'>ConversoAI – PDF Conversational RAG</div>")
    gr.Markdown("<div id='app-subtitle'>Upload PDFs, build a knowledge base, and chat with them using Groq ✨</div>")

    with gr.Row():
        with gr.Column(scale=1):
            api_key = gr.Textbox(
                label="🔑 Enter your Groq API Key",
                type="password",
                placeholder="sk_...",
            )
            session_id = gr.Textbox(
                label="🧾 Session ID",
                value="default_session",
                info="Use a different ID to keep multiple independent conversations.",
            )
            pdf_files = gr.File(
                label="📚 Upload your PDF files",
                file_types=[".pdf"],
                file_count="multiple",
            )
            build_btn = gr.Button("🚀 Build Knowledge Base", variant="primary")
            status_md = gr.Markdown("➡️ Start by entering your Groq key, uploading PDFs and then click the button.")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="💬 Chat with your PDFs",
                height=430,
            )
            user_question = gr.Textbox(
                label="Ask a question about your PDFs",
                placeholder="e.g., What are the main conclusions of chapter 2?",
            )
            ask_btn = gr.Button("✨ Ask", variant="primary")

    # Hidden state: the RAG chain object
    chain_state = gr.State(value=None)

    # When user clicks "Build Knowledge Base"
    def on_build_clicked(api_key, files, session_id):
        chain, msg = setup_rag_pipeline(api_key, files, session_id)
        return chain, msg

    build_btn.click(
        on_build_clicked,
        inputs=[api_key, pdf_files, session_id],
        outputs=[chain_state, status_md],
    )

    # When user asks a question
    ask_btn.click(
        answer_question,
        inputs=[user_question, session_id, chain_state, chatbot],
        outputs=[chatbot],
    )

    user_question.submit(
        answer_question,
        inputs=[user_question, session_id, chain_state, chatbot],
        outputs=[chatbot],
    )

if __name__ == "__main__":
    demo.launch()
