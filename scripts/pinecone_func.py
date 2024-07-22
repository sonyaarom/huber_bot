def flatten_values(documents):
    for doc in documents:
        if isinstance(doc['values'][0], list):
            doc['values'] = doc['values'][0]
            
    return documents

def pinecone_upsert(index, documents, batch_size=1000):
    documents = flatten_values(documents)
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        index.upsert(vectors=batch)



