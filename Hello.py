import streamlit as st
import os
from tempfile import NamedTemporaryFile
from pdf_extractor import PDFExtractor
from inference_utils import *

#os.system('pip install -U flash_attn==2.6.1 --no-build-isolation')

st.title("LLM-Powered Document Chat")


with st.sidebar.form(key='docs_form', clear_on_submit=False):
    st.write("Upload your PDF documents here:")
    documents = st.file_uploader('Documents', type=['pdf'], accept_multiple_files=True, key='doc_pdf', help=None, label_visibility="visible")
    hf_api_token = st.text_input("Enter your HuggingFace access token/API key:", type='password')
    submitted = st.form_submit_button(label="Submit", help=None, on_click=None, type="secondary", disabled=False)
    if submitted:
        st.session_state.documents = documents
        st.session_state.hf_api_token = hf_api_token
        path = "./pdf_images"
        st.session_state.all_texts, st.session_state.all_tables = [], []
        for doc in documents:
            with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
                f.write(doc.getbuffer())
                obj = PDFExtractor(path, f.name)
                texts, tables = obj.categorize_elements()
                st.session_state.all_texts.extend(texts)
                st.session_state.all_tables.extend(tables)
all_texts, all_tables = st.session_state.all_texts, st.session_state.all_tables
st.session_state.all_texts, st.session_state.all_tables = [], []

with st.form(key='inference_form', clear_on_submit=False):
    question = st.text_input('Question: ')
    submitted = st.form_submit_button(label="Submit", help=None, on_click=None, type="secondary", disabled=False)
    if submitted:
        docs = split_text(all_texts)
        rag_extracts = get_rag_hits(docs, 'cross_encoder', question)
        response = infer_query(question, rag_extracts, hf_api_token, model_id="meta-llama/Meta-Llama-3-8B-Instruct")
        st.write(response)

