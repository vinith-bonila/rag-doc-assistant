# Retrieval-Augmented Generation (RAG) Concepts

## What RAG Solves
Large language models answer from parameters fixed at training time, so they go
stale and can hallucinate on private or recent data. Retrieval-Augmented
Generation fixes this by fetching relevant passages from an external document
store at query time and grounding the model's answer in that retrieved text.

## Chunking
Documents are split into chunks before indexing. Chunk size is a trade-off:
chunks that are too large dilute the embedding and retrieve irrelevant text,
while chunks that are too small lose context. Semantic chunking splits at points
where the topic shifts rather than at fixed character counts, keeping coherent
ideas together.

## Embeddings and Vector Search
A bi-encoder embedding model maps each chunk to a dense vector. Vectors are
stored in a vector database such as FAISS, Milvus, or Pinecone. At query time
the question is embedded and the nearest chunk vectors are retrieved by cosine
similarity. FAISS with an inner-product index over normalized vectors is a fast,
common choice for this.

## Reranking
Bi-encoder retrieval is fast but scores the query and passage separately. A
cross-encoder reranker reads the query and passage together and produces a much
more accurate relevance score. The standard two-stage pattern retrieves many
candidates with the bi-encoder, then reranks a small shortlist with the
cross-encoder before sending the best passages to the LLM.

## Grounding and Citations
To keep answers trustworthy, the LLM is instructed to answer only from the
retrieved context, to cite which passages it used, and to admit when the context
does not contain the answer. This reduces hallucination and makes every claim
auditable.

## Evaluation
Retrieval is measured with recall@k: whether the correct source appears in the
top-k results. Answer quality is measured with faithfulness, which checks whether
the answer's claims are supported by the retrieved context. Together they
separate retrieval failures from generation failures.
