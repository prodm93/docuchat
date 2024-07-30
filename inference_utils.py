__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
from huggingface_hub import InferenceClient
from transformers import AutoTokenizer, TextStreamer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
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
    db = Chroma.from_documents(docs, embeddings)
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
    client = InferenceClient(token=hf_api_key)
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

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_api_key)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = client.text_generation(
        prompt=prompt,
        model=model_id,
        temperature=0.1,
        max_new_tokens=512,
        seed=42,
        return_full_text=False
    )
    
    return response


def infer_query_chatbot(question, rag_extracts, conversation_history, hf_api_key, model_id="meta-llama/Meta-Llama-3-8B-Instruct"):

    client = InferenceClient(token=hf_api_key)
    system_input = """You are an assistant chatbot tasked with answering questions from complex documents
        using retrieval-augmented generation (RAG). Answer the user question as accurately as possible
        based on the query hits extracted using RAG, which are both provided by the user. Incorporate information
        from the conversation history, but only if it is relevant to the current question. If you do not know 
        the answer, respond with 'I'm sorry, but I am unable to find an answer to that question.' Do not make references 
        to extract numbers in your response."""

    user_input = f"""Using the RAG-generated extracts--as well as the conversation history, if relevant--below, please answer my question:
        Question: {question}
        Extracts: {rag_extracts}
        Conversation history: {conversation_history}"""

    messages = [
        {"role": "system", "content": system_input},
        {"role": "user", "content": user_input},
    ]

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_api_key)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    """text_streamer = TextStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )"""
    
    terminators = [tokenizer.eos_token_id,
                       tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    """outputs = model.generate(
            input_ids,
            max_new_tokens=512,  # Adjust as needed
            eos_token_id=terminators,
            temperature=0.1,
            do_sample=True,
            streamer=text_streamer
        )
    
    output_text = tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)"""
    output_text = client.text_generation(
        prompt=prompt,
        model=model_id,
        temperature=0.1,
        #repetition_penalty=0.1,
        max_new_tokens=512,
        do_sample=True,
        stop_sequences=['<|eot_id|>', tokenizer.eos_token],
        #skip_prompt=True,
        #skip_special_tokens=True,
        return_full_text=False,
        stream=True
    )


    return output_text

