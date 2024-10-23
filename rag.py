import os
import pandas as pd
import torch
import argparse
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain.schema import BaseRetriever
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain.schema import Document, BaseRetriever
from transformers import AutoModelForSequenceClassification, AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration

import warnings
warnings.filterwarnings("ignore")

K = 5

def load_and_split_data(directory_path: str) -> List[str]:
    # Split the data into chunks of 1000 characters with 100 characters overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = []
    for file in os.listdir(directory_path):
        if file.endswith(".txt"):
            with open(os.path.join(directory_path, file), 'r') as f:
                text = f.read()
                docs.extend(text_splitter.split_text(text))
    return docs

def setup_vector_store(context_docs: List[str], force_new: bool = False, cache_dir: str = "./vector_cache"):
    # Load the embeddings model and create FAISS embeddings
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    index_path = "faiss_index"
    if os.path.exists(index_path) and not force_new:
        print("Loading existing vector store from cache...")
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Creating new vector store...")
        vector_store = FAISS.from_texts(context_docs, embeddings)
        vector_store.save_local(index_path)
        return vector_store

def setup_reranker():
    # Load the reranker model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    return model, tokenizer

def rerank_documents(query: str, docs: List[str], model, tokenizer, top_k: int = K) -> List[str]:
    # Rerank the documents using the reranker model
    pairs = [[query, doc] for doc in docs]
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
    
    with torch.no_grad():
        scores = model(**inputs).logits.squeeze()
    
    ranked_results = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked_results[:top_k]]

def setup_query_expansion():
    # Load the query expansion tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("BeIR/query-gen-msmarco-t5-base-v1", legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("BeIR/query-gen-msmarco-t5-base-v1")
    return tokenizer, model

def expand_query(query: str, tokenizer, model, num_expansions: int = 3) -> List[str]:
    # Expand the query using the query expansion model
    input_ids = tokenizer.encode(f"Expand the query: {query}", return_tensors="pt", max_length=64, truncation=True)
    outputs = model.generate(
        input_ids=input_ids,
        max_length=64,
        num_return_sequences=num_expansions,
        num_beams=num_expansions,
        no_repeat_ngram_size=2
    )
    expanded_queries = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return [query] + expanded_queries

def setup_retrievers(context_docs: List[str], vector_store):
    # Setup the BM25 and FAISS retrievers
    bm25_retriever = BM25Retriever.from_texts(context_docs)
    bm25_retriever.k = K
    faiss_retriever = vector_store.as_retriever(search_kwargs={"k": K})
    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.6, 0.4]
    )
    
def improved_retrieval(query: str, ensemble_retriever, query_expansion_tokenizer, query_expansion_model, reranker_model, reranker_tokenizer, top_k: int = 5) -> List[Document]:
    # Perform the improved retrieval process
    expanded_queries = expand_query(query, query_expansion_tokenizer, query_expansion_model)
    all_docs = []
    for q in expanded_queries:
        docs = ensemble_retriever.get_relevant_documents(q)
        all_docs.extend([doc.page_content for doc in docs])
    unique_docs = list(set(all_docs))
    reranked_docs = rerank_documents(query, unique_docs, reranker_model, reranker_tokenizer, top_k)
    return [Document(page_content=doc) for doc in reranked_docs]


def setup_qa_chain(improved_retriever, approach):
    # Setup the QA chain
    llama_llm = OllamaLLM(model="llama3.2")
    
    if approach == "zero-shot":
        prompt_template = """Answer the user question based on the context provided below
                Context : {context}
                Question: {question}
                Keep your answer concise and to the point.
                """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    else:
        # Setup the few-shot examples
        few_shot_examples = [
            {"question": "What is the name of the annual pickle festival held in Pittsburgh?", "answer": "Picklesburgh"},
            {"question": "When was the Pittsburgh Soul Food Festival established?", "answer": "2019"},
            {"question": "Who is performing the Opera event on 7th November?", "answer": "Cavalleria Rusticana and Pagliacci"},
            {"question": "When is the Syracuse Orange vs. Robert Morris Colonials women's ice hockey game scheduled?", "answer": "February 8, 2025"},
            {"question": "How many Super Bowls have the Pittsburgh Steelers won?", "answer": "Six"},
            {"question": "Where is the \"Like, Totally Transformative: CMU in the 1980s\" exhibit being held?", "answer": "Hunt Library Gallery"},
        ]

        example_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="Question: {question}\nAnswer: {answer}"
        )

        prompt = FewShotPromptTemplate(
            examples=few_shot_examples,
            example_prompt=example_prompt,
            prefix="Answer the following question based on the context provided. Keep the answer as short as possible. Do not include any other information. Here are some examples:",
            suffix="Context: {context}\nQuestion: {question}\nAnswer:",
            input_variables=["context", "question"],
            example_separator="\n\n"
        )

    # Setup the RetrievalQA chain
    return RetrievalQA.from_chain_type(
        llm=llama_llm,
        chain_type="stuff",
        retriever=improved_retriever,
        chain_type_kwargs={"prompt": prompt}
    )

def get_answer(qa_chain, question):
    # Get the answer from the QA chain
    result = qa_chain.invoke({"query": question})
    return result["result"]

def process_file(input_file, output_file, qa_chain):
    # Process the input file and write the answers to the output file
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            question = line.strip()
            answer = get_answer(qa_chain, question)
            outfile.write(f"{answer}\n")

def setup_retrieval_pipeline(data_dir, approach, force_new_vector_store):
    # Setup the retrieval pipeline
    context_docs = load_and_split_data(data_dir)
    vector_store = setup_vector_store(context_docs, force_new=force_new_vector_store)
    reranker_model, reranker_tokenizer = setup_reranker()
    query_expansion_tokenizer, query_expansion_model = setup_query_expansion()
    ensemble_retriever = setup_retrievers(context_docs, vector_store)
    
    class CustomRetriever(BaseRetriever):
        # Perform the improved retrieval process
        def _get_relevant_documents(self, query: str) -> List[Document]:
            return improved_retrieval(
                query,
                ensemble_retriever,
                query_expansion_tokenizer,
                query_expansion_model,
                reranker_model,
                reranker_tokenizer
            )
    
    custom_retriever = CustomRetriever()
    return setup_qa_chain(custom_retriever, approach)


def main():
    parser = argparse.ArgumentParser(description="QA system for Pittsburgh-related questions")
    parser.add_argument("--data_dir", required=True, help="Directory containing the data files")
    parser.add_argument("--input", help="Input text file with questions")
    parser.add_argument("--output", help="Output text file for answers")
    parser.add_argument("--query", help="Single question to answer")
    parser.add_argument("--approach", choices=["zero-shot", "few-shot"], default="few-shot", help="Choose between zero-shot and few-shot approaches")
    parser.add_argument("--force_new_vector_store", action="store_true", help="Force creation of a new vector store instead of using cached version")
    
    args = parser.parse_args()
    
    qa_chain = setup_retrieval_pipeline(args.data_dir, args.approach, args.force_new_vector_store)
    
    print("Retrieval pipeline setup complete.")

    if args.input and args.output:
        process_file(args.input, args.output, qa_chain)
    elif args.query:
        answer = get_answer(qa_chain, args.query)
        print(f"Question: {args.query}")
        print(f"Answer: {answer}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()