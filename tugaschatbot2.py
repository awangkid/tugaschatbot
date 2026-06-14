import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Matikan file watcher biar log bersih (opsional)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# === API KEY Groq langsung di kode ===
GROQ_API_KEY = "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"

st.title("📚 Chatbot Multi-PDF dengan Groq + Strict Mode. Chat tergantung dengan PDF yang diinput")

uploaded_files = st.file_uploader("Upload file PDF (bisa lebih dari satu)", type="pdf", accept_multiple_files=True)

# Cache supaya proses build index hanya dilakukan sekali per kumpulan file
@st.cache_resource(show_spinner=False)
def build_qa_chain(files_data):
    docs = []
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for file_name, file_bytes in files_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        for doc in documents:
            doc.metadata["source"] = file_name

        docs.extend(text_splitter.split_documents(documents))
        os.remove(tmp_path)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()

    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant", temperature=0)

    prompt_template = """
    Gunakan hanya informasi dari konteks berikut untuk menjawab.
    Jika pertanyaan tidak relevan dengan konteks, jawab:
    "Maaf, pertanyaan ini di luar cakupan chatbot PDF."

    Konteks:
    {context}

    Pertanyaan: {question}
    Jawaban:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    return qa, len(docs)


if uploaded_files:
    # Siapkan data file (nama + bytes) untuk dijadikan key cache
    files_data = tuple((f.name, f.getvalue()) for f in uploaded_files)

    with st.spinner("Memproses PDF dan membangun index (hanya sekali)..."):
        try:
            qa, total_chunks = build_qa_chain(files_data)
            st.success(f"✅ Index siap! Total {total_chunks} chunk dari {len(uploaded_files)} file.")
        except Exception as e:
            st.error(f"Terjadi error: {e}")
            st.stop()

    # Inisialisasi chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Tampilkan riwayat percakapan
    for chat in st.session_state.history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            with st.expander("Sumber"):
                for src in chat["sources"]:
                    st.write(f"- {src}")

    # Input chat (bisa terus dipakai berkali-kali)
    query = st.chat_input("Tanyakan sesuatu dari PDF:")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Mencari jawaban..."):
                result = qa.invoke({"query": query})
                answer = result["result"]

                sources = []
                for doc in result["source_documents"]:
                    sources.append(f"{doc.metadata.get('source', 'Unknown')} - Halaman {doc.metadata.get('page', 'N/A')}")

                st.write(answer)
                with st.expander("Sumber"):
                    for src in sources:
                        st.write(f"- {src}")

        st.session_state.history.append({"question": query, "answer": answer, "sources": sources})

else:
    st.info("Silakan upload file PDF terlebih dahulu.")