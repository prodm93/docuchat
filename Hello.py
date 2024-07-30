import streamlit as st
import os
#from tempfile import NamedTemporaryFile
import tempfile
from pdf_extractor import PDFExtractor
from inference_utils import *


st.markdown(
    """
<style>
    .st-emotion-cache-janbn0 {
        flex-direction: row-reverse;
        text-align: right;
    }
</style>
""",
    unsafe_allow_html=True,
)

#os.system('pip install -U flash_attn==2.6.1 --no-build-isolation')

st.title("LLM-Powered RAG Document Chat")

with st.sidebar.form(key='docs_form', clear_on_submit=False):
    st.write("Upload your PDF documents here:")
    documents = st.file_uploader('Documents', type=['pdf'], accept_multiple_files=True, key='doc_pdf', help=None, label_visibility="visible")
    hf_api_token = st.text_input("Enter your HuggingFace access token/API key:", type='password')
    submitted = st.form_submit_button(label="Submit", help=None, on_click=None, type="secondary", disabled=False)
    if submitted:
        st.session_state.documents = documents
        st.session_state.hf_api_token = hf_api_token
        img_path = "./pdf_images"
        all_texts, all_tables = [], []
        for doc in documents:
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, doc.name)
            with open(file_path, 'wb') as f:
                f.write(doc.getvalue())
                obj = PDFExtractor(img_path, file_path)
                texts, tables = obj.categorize_elements()
                all_texts.extend(texts)
                all_tables.extend(tables)
        st.session_state.all_texts = all_texts
        st.session_state.all_tables = all_tables
        st.session_state.docs = split_text(st.session_state.all_texts)

#with st.form(key='inference_form', clear_on_submit=False):
    #question = st.text_input('Question: ')
    #submitted = st.form_submit_button(label="Submit", help=None, on_click=None, type="primary", disabled=False)
    #if submitted:
        #rag_extracts = get_rag_hits(st.session_state.docs, 'cross_encoder', question)
        #response = infer_query(question, rag_extracts, hf_api_token, model_id="meta-llama/Meta-Llama-3-8B-Instruct")
        #st.write(response)

def main():

    #st.title("LLM-Powered RAG Document Query (Chatbot Version)")
        # Button to clear chat history
    def clear_chat_history():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey there, how may I assist you today?"}]

    #st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    st.button('Clear Chat History', on_click=clear_chat_history)
    
    # Initialize chat messages session state if not present
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey there, how may I assist you today?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input for a chat prompt
    if question := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

    # Generate  response if the last message is not from the assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st.write(st.session_state.messages)
                rag_extracts = get_rag_hits(st.session_state.docs, 'cross_encoder', question)
                response = infer_query_chatbot(question, rag_extracts, st.session_state.hf_api_token, 
                                               model_id="meta-llama/Meta-Llama-3-8B-Instruct")
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)

        # Store the assistant's response in the chat messages
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)


if __name__ == '__main__':
    main()


