from langchain_community.vectorstores import Chroma  # type: ignore
from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore


def inspect_chroma_db(db_path: str):

    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Charge la base existante
    db = Chroma(persist_directory=db_path, embedding_function=embedding_function)

    # Récupère la collection interne
    collection = db._collection
    results = collection.get(include=["documents", "embeddings", "metadatas"])

    for i in range(len(results["ids"])):
        print(f"ID: {results['ids'][i]}")
        print(f"Document: {results['documents'][i]}")
        print(f"Embedding (tronc.): {results['embeddings'][i][:5]}...")
        print(f"Metadata: {results['metadatas'][i]}")
        print("----")



# inspect_chroma_db("./chroma_db_mistral")
