# Upload documents in batches
def pinecone_upsert(index, documents, batch_size=1000):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        index.upsert(vectors=batch)


