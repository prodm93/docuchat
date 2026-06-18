__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import nltk
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng')

import os
from huggingface_hub import InferenceClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.docstore.document import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank, CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def split_text(all_texts, chunk_size=128, chunk_overlap=10):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents([Document(page_content=x) for x in all_texts])
    return docs


def get_rag_hits(docs, rerank_method, question: str):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Qdrant.from_documents(docs, embeddings, location=":memory:", collection_name="docs")
    k_val = len(docs)//50 if len(docs) > 250 else 7
    dense_retriever = db.as_retriever(search_kwargs = {"k":k_val})
    sparse_retriever = BM25Retriever.from_documents(docs)
    sparse_retriever.k = k_val
    ensemble_retriever = EnsembleRetriever(retrievers = [dense_retriever, sparse_retriever],
                                                        weights = [0.3, 0.7])
    if rerank_method == 'flash_rerank': 
        FlashrankRerank.update_forward_refs()
        compressor = FlashrankRerank(top_n=k_val)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
    elif rerank_method == 'cross_encoder':
        hf_rerank_model = HuggingFaceCrossEncoder(model_name='mixedbread-ai/mxbai-rerank-xsmall-v1')
        compressor = CrossEncoderReranker(model=hf_rerank_model, top_n=k_val)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )
    matches = compression_retriever.invoke(question)
    rag_extracts = "\n\nExtract:\n" + "\n\nExtract:\n".join([doc.page_content for doc in matches])
    return rag_extracts


def infer_query(question, rag_extracts, hf_api_key, model_id="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    client = InferenceClient(token=hf_api_key, timeout=600)
    system_input = """You are an assistant tasked with answering questions from complex documents
        using retrieval-augmented generation (RAG). Answer the user question as accurately as possible
        based on the query hits extracted using RAG, which are both provided by the user. If you do not know the answer, 
        respond with 'I'm sorry, but I am unable to find an answer to that question.'"""

    user_input = f"""Using the RAG-generated extracts below, please answer my question:
        Question: {question}
        Extracts: {rag_extracts}"""

    messages = [
        {"role": "system", "content": system_input},
        {"role": "user", "content": user_input},
    ]

    response = client.chat_completion(
        model=model_id,
        messages=messages,
        temperature=0.1,
        max_tokens=512,
    )
    return response.choices[0].message.content
    

def infer_query_chatbot(question, rag_extracts, conversation_history, hf_api_key, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):

    client = InferenceClient(token=hf_api_key, timeout=600)
    system_input = """You are an assistant chatbot tasked with answering questions from complex documents
        using retrieval-augmented generation (RAG). Answer the user question as accurately as possible
        based on the query hits extracted using RAG, which are both provided by the user. Incorporate information
        from the conversation history, but only if it is relevant to the current question. If you do not know 
        the answer, respond with 'I'm sorry, but I am unable to find an answer to that question.' Do not make references 
        to extract numbers or mention the term 'conversation history' in your response."""

    user_input = f"""Using the RAG-generated extracts--as well as the conversation history, if relevant--below, please answer my question:
        Question: {question}
        Extracts: {rag_extracts}
        Conversation history: {conversation_history}"""

    messages = [
        {"role": "system", "content": system_input},
        {"role": "user", "content": user_input},
    ]
    
    for chunk in client.chat_completion(
        model=model_id,
        messages=messages,
        temperature=0.1,
        max_tokens=512,
        stream=True,
    ):
        content = chunk.choices[0].delta.content
        if content:
            yield content

