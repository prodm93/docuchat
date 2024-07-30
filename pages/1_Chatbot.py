import streamlit as st
from inference_utils import *

st.set_page_config(page_title="LLM-Powered RAG Document Query (Chatbot Version)")


def main():

    st.title("LLM-Powered RAG Document Query (Chatbot Version)")
 
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

    # Button to clear chat history
    def clear_chat_history():
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I assist you today?"}]

    #st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    st.button('Clear Chat History', on_click=clear_chat_history)


if __name__ == '__main__':
    main()
