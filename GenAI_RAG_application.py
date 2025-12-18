import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableSequence
import tempfile

load_dotenv()
embedding_model = OpenAIEmbeddings()
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

st.set_page_config(page_title="Q&A For Your File", layout="wide")
st.title("Chat With Your File")

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

uploaded_file = st.file_uploader("Upload Your File")

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1] 
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    st.info("📑 Processing File...")

    loader = UnstructuredFileLoader(file_path, encoding="utf-8")
    # loader = TextLoader(file_path, encoding="utf-8")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,           
        chunk_overlap=200,         
        length_function=len,      
        is_separator_regex=False,  
        separators=["\n\n", "\n", " ", ""] 
    )
    chunks = text_splitter.split_documents(docs)

    st.session_state.vector_db = FAISS.from_documents(chunks, embedding_model)

    st.success("✅ PDF processed. You can now ask questions!")

def build_chain(vector_db):

    retriever = vector_db.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": 5, "lambda_mult": 0.4}
    )

    def related_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided context below.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables=['context', 'question']
    )

    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(related_docs),
        "question": RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | model | parser

    return final_chain

if st.session_state.vector_db:
    query = st.text_input("🔎 Ask a question about your File:")

    if query:
        with st.spinner("Thinking..."):
            chain = build_chain(st.session_state.vector_db)
            result = chain.invoke(query)

        st.subheader("💡 Answer")
        st.write(result)



















# multi_query_retriever = MultiQueryRetriever.from_llm(
#     retriever=retriever,
#     llm=model
# )


