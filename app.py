from dotenv import load_dotenv
import streamlit as st
import time
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_community.callbacks import get_openai_callback
from langchain_openai import OpenAIEmbeddings
# Sidebar contents
with st.sidebar:
    st.title('💬PDF Summarizer and Q/A App')
    st.markdown('''
    ## About this application
    You can built your own customized LLM-powered chatbot using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(2)
    st.write(' Why drown in papers when your chat buddy can give you the highlights and summary? Happy Reading. ')
    add_vertical_space(2)    
    st.write('Made by ***Sangita Pokhrel***')
def process_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return text, chunks

def create_knowledge_base(chunks):
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

knowledge_base = None
old_pdf = None
def main():
    load_dotenv()
    global knowledge_base, old_pdf
    #Main Content
    st.header("Ask About Your PDF 🤷‍♀️💬")


    # upload file
    pdf = st.file_uploader("Upload your PDF File and Ask Questions", type="pdf")
    
    # extract the text
    if pdf is not None:
      if knowledge_base is None:
        pdf_reader = PdfReader(pdf)
        text, chunks = process_pdf(pdf)
        knowledge_base = create_knowledge_base(chunks)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        old_pdf = pdf
      else:
        if pdf != old_pdf:
           knowledge_base = None
      if knowledge_base:
        # show user input
        with st.chat_message("user"):
          st.write("Hello World 👋")
        user_question = st.text_input("Please ask a question about your PDF here:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            with get_openai_callback() as cb:
                response = chain.invoke( {"input_documents":docs, "question": user_question} )
                print(cb)

            st.write(response)

    

if __name__ == '__main__':
    main()
