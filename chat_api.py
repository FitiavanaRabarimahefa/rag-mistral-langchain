from fastapi import FastAPI  # type: ignore
from pydantic import BaseModel  # type: ignore
from typing import List, Optional  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
import uvicorn  # type: ignore

from langchain.chains import LLMChain  # type: ignore

from bot import (
    calculate_relevance_score,
    format_docs_for_prompt,
    load_jsonl_documents,
    setup_mistral_llm,
    setup_reranker,
    setup_vectordb,
    trim_context,
    rerank_with_cross_encoder,
    ANSWER_PROMPT
)


# Initialisation FastAPI

app = FastAPI(title="Chatbot RAG Mistral", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Schémas pour la requête/réponse

class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: Optional[List[str]] = []
    relevance_score: float


# -----------------------------
# Initialisation globale (chargement modèles/db)
# -----------------------------
print("Initialisation du Chatbot RAG Mistral...\n")

docs = load_jsonl_documents()
if not docs:
    raise RuntimeError("Aucun document trouvé. Vérifiez ./docs/knowledge_structured.jsonl")

vectordb, embedding_model = setup_vectordb(docs)
if vectordb is None:
    raise RuntimeError("Impossible de créer la base vectorielle")

llm = setup_mistral_llm()
if not llm:
    raise RuntimeError("Impossible de charger le modèle Mistral")

reranker = setup_reranker()
answer_chain = LLMChain(llm=llm, prompt=ANSWER_PROMPT)

retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 8, "fetch_k": 32, "lambda_mult": 0.7}
)

RELEVANCE_THRESHOLD = 0.25

@app.get("/")
def hello():
    return {"message": "Bienvenue sur le Chatbot RAG Mistral. Utilisez le endpoint /chat pour poser des questions."}

# Endpoint principal


'''
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    query = request.question.strip()
    if not query:
        return ChatResponse(answer="Veuillez poser une question.", sources=[], relevance_score=0.0)

    candidates = retriever.get_relevant_documents(query)
    if not candidates:
        return ChatResponse(answer="Je ne trouve pas d'informations pertinentes.", sources=[], relevance_score=0.0)

    # Re-ranking si dispo
    if reranker:
        candidates, scores = rerank_with_cross_encoder(query, candidates, reranker, top_k=5)
        if not candidates or max(scores) < 0.3:
            return ChatResponse(answer="Aucune information pertinente.", sources=[], relevance_score=0.0)

    candidates = trim_context(candidates, max_chars=6000)
    relevance_score = calculate_relevance_score(query, candidates, embedding_model)
    if relevance_score < RELEVANCE_THRESHOLD:
        return ChatResponse(answer="Aucune information suffisamment pertinente.", sources=[], relevance_score=relevance_score)

    context_text = format_docs_for_prompt(candidates)
    out = answer_chain.invoke({"context": context_text, "input": query})
    answer = (out.get("text") or "").strip()

    if not answer or len(answer) < 10:
        answer = "Je ne trouve pas d'informations pertinentes dans ma base de connaissances."

    return ChatResponse(
        answer=answer,
        sources=[doc.metadata.get("source", "inconnu") for doc in candidates],
        relevance_score=relevance_score
    )
'''
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    query = request.question.strip()
    if not query:
        return ChatResponse(answer="Veuillez poser une question.", sources=[], relevance_score=0.0)

    candidates = retriever.get_relevant_documents(query)
    if not candidates:
        return ChatResponse(answer="Je ne trouve pas d'informations pertinentes.", sources=[], relevance_score=0.0)

    # Re-ranking
    if reranker:
        candidates, scores = rerank_with_cross_encoder(query, candidates, reranker, top_k=5)
        if not candidates or max(scores) < 0.3:
            return ChatResponse(answer="Aucune information pertinente.", sources=[], relevance_score=0.0)

    candidates = trim_context(candidates, max_chars=6000)
    relevance_score = calculate_relevance_score(query, candidates, embedding_model)
    if relevance_score < RELEVANCE_THRESHOLD:
        return ChatResponse(answer="Aucune information suffisamment pertinente.", sources=[], relevance_score=relevance_score)

    context_text = format_docs_for_prompt(candidates)
    out = answer_chain.invoke({"context": context_text, "input": query})
    raw_answer = (out.get("text") or "").strip()

    # Extraire uniquement la partie après "Réponse :"
    answer = raw_answer
    if "Réponse :" in raw_answer:
        answer = raw_answer.split("Réponse :", 1)[1].strip()
        answer = answer  # on garde le format demandé

    if not answer or len(answer) < 5:
        answer = "Je ne trouve pas d'informations pertinentes dans ma base de connaissances."

    return ChatResponse(
        answer=answer,
        sources=[doc.metadata.get("source", "inconnu") for doc in candidates],
        relevance_score=relevance_score
    )




if __name__ == "__main__":
    uvicorn.run("chat:app", host="0.0.0.0", port=8000, reload=True)
