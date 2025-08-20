from langchain.schema import Document  # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline  # type: ignore
from langchain_community.vectorstores import Chroma  # type: ignore
from langchain.prompts import PromptTemplate  # type: ignore
from langchain.chains import LLMChain  # type: ignore
from dotenv import load_dotenv  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore
import torch  # type: ignore
import json
import numpy as np  # type: ignore
import os
from sentence_transformers import CrossEncoder  # type: ignore
import re

load_dotenv()



# Utils JSONL / Métadonnées
def flatten_dict(d, parent_key='', sep='_'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        elif isinstance(v, list):
            items[new_key] = ", ".join(map(str, v))
        else:
            items[new_key] = v
    return items


def load_jsonl_documents():
    docs = []
    jsonl_path = "./docs/knowledge_structured.jsonl"

    if not os.path.exists(jsonl_path):
        print(f"Fichier JSONL non trouvé : {jsonl_path}")
        return docs

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    contenu = data.get("contenu", "").strip()
                    if not contenu:
                        continue

                    metadata = {
                        "id_chunk": data.get("id_chunk"),
                        "id_document": data.get("id_document"),
                    }

                    for key in ["metadonnees", "infos_chunk", "donnees_structurees",
                                "relations", "contexte", "metriques_qualite"]:
                        if key in data and isinstance(data[key], dict):
                            metadata.update(flatten_dict(data[key]))

                    docs.append(Document(page_content=contenu, metadata=metadata))
                except json.JSONDecodeError:
                    print(f"Ligne {line_num} ignorée (JSON invalide) : {line[:120]}")
        print(f"✓ {len(docs)} documents JSONL chargés.")
    except Exception as e:
        print(f"Erreur lors du chargement JSONL : {e}")
    return docs



# Base vectorielle
def setup_vectordb(docs, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    if not docs:
        print("Aucun document à indexer.")
        return None, None
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = splitter.split_documents(docs)
        print(f"✓ Documents divisés en {len(split_docs)} chunks.")

        embedding = HuggingFaceEmbeddings(model_name=embedding_model)
        print(f"✓ Modèle d'embedding '{embedding_model}' chargé.")

        vectordb = Chroma.from_documents(
            split_docs,
            embedding=embedding,
            persist_directory="./chroma_db_mistral"
        )
        print("✓ Base de données vectorielle créée.")
        return vectordb, embedding
    except Exception as e:
        print(f"Erreur lors de la création de la base vectorielle : {e}")
        return None, None



# Modèle Mistral
def setup_mistral_llm():
    MODEL_PATH = "./mistral-7b-instruct-local"
    if not os.path.exists(MODEL_PATH):
        print(f"Modèle Mistral non trouvé : {MODEL_PATH}")
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

        gen_kwargs = dict(
            max_new_tokens=384,
            temperature=0.1,
            top_p=1.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, **gen_kwargs)
        llm = HuggingFacePipeline(pipeline=pipe)
        print("✓ Modèle Mistral local chargé.")
        return llm
    except Exception as e:
        print(f"Erreur lors du chargement de Mistral : {e}")
        return None



# Reranker
def setup_reranker():
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✓ Reranker cross-encoder chargé.")
        return reranker
    except Exception as e:
        print(f"Erreur lors du chargement du reranker : {e}")
        return None


def rerank_with_cross_encoder(query, docs, reranker, top_k=5):
    if not docs:
        return [], []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    order = np.argsort(scores)[::-1]
    top_docs = [docs[i] for i in order[:top_k]]
    top_scores = [float(scores[i]) for i in order[:top_k]]
    return top_docs, top_scores



# Pertinence
def calculate_relevance_score(query, docs, embedding_model):
    if not docs:
        return 0.0
    try:
        q_emb = embedding_model.embed_query(query)
        d_embs = embedding_model.embed_documents([d.page_content for d in docs])
        sims = [float(np.dot(q_emb, e) / ((np.linalg.norm(q_emb) * np.linalg.norm(e)) or 1e-9)) for e in d_embs]
        return float(np.mean(sims)) if sims else 0.0
    except Exception as e:
        print(f"Erreur pertinence : {e}")
        return 0.0


def trim_context(docs, max_chars=6000):
    out, total = [], 0
    for d in docs:
        c = d.page_content
        if total + len(c) <= max_chars:
            out.append(d)
            total += len(c)
        else:
            remain = max_chars - total
            if remain > 200:
                out.append(Document(page_content=c[:remain], metadata=d.metadata))
            break
    return out


def format_docs_for_prompt(docs):
    lines = []
    for d in docs:
        id_doc = d.metadata.get("id_document")
        id_chunk = d.metadata.get("id_chunk")
        title = d.metadata.get("metadonnees_nom_entite") or d.metadata.get("nom_entite") or ""
        lines.append(f"[id_document={id_doc} id_chunk={id_chunk} title='{title}']\n{d.page_content}")
    return "\n\n".join(lines)



# Prompts
ANSWER_PROMPT = PromptTemplate.from_template(
    """Tu es un assistant IA et tu réponds STRICTEMENT en français.
Si la réponse n'est pas dans le contexte, répond exactement :
"Je ne trouve pas d'informations pertinentes dans ma base de connaissances pour répondre à cette question."

Contexte:
{context}

Question: {input}

Réponse :"""
)


def print_sources(docs):
    seen = set()
    for d in docs:
        sid = (d.metadata.get("id_document"), d.metadata.get("id_chunk"))
        if sid in seen:
            continue
        seen.add(sid)
        title = d.metadata.get("metadonnees_nom_entite") or d.metadata.get("nom_entite") or ""
        src = d.metadata.get("metadonnees_source") or d.metadata.get("source") or ""
        print(f" - source id_document={sid[0]} id_chunk={sid[1]} title='{title}' source='{src}'")



# Main chatbot
def main():
    print("Initialisation du Chatbot RAG Mistral...\n")

    docs = load_jsonl_documents()
    if not docs:
        print("Aucun document trouvé. Vérifiez ./docs/knowledge_structured.jsonl")
        return

    vectordb, embedding_model = setup_vectordb(docs)
    if vectordb is None:
        return

    llm = setup_mistral_llm()
    if not llm:
        print("Impossible de charger le modèle Mistral.")
        return

    reranker = setup_reranker()
    if not reranker:
        print(" Reranker indisponible. On continuera sans reranking.")

    answer_chain = LLMChain(llm=llm, prompt=ANSWER_PROMPT)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 32, "lambda_mult": 0.7})

    chat_history = []
    RELEVANCE_THRESHOLD = 0.25

    print("Chatbot prêt ! Tapez 'exit' pour quitter.")
    print("-" * 50)

    while True:
        try:
            query = input("\n Vous : ").strip()
            if query.lower() in {"exit", "quit", "q", "sortir"}:
                print("\nAu revoir !")
                break
            if not query:
                continue

            candidates = retriever.get_relevant_documents(query)
            if not candidates:
                print("Réponse : Je ne trouve pas d'informations pertinentes dans ma base de connaissances pour répondre à cette question.")
                continue

            # Re-ranking strict
            if reranker:
                candidates, scores = rerank_with_cross_encoder(query, candidates, reranker, top_k=5)
                if not candidates or max(scores) < 0.3:
                    print("Réponse : Aucune information pertinente.")
                    continue

            candidates = trim_context(candidates, max_chars=6000)
            relevance_score = calculate_relevance_score(query, candidates, embedding_model)
            if relevance_score < RELEVANCE_THRESHOLD:
                print("Réponse : Aucune information suffisamment pertinente.")
                continue

            context_text = format_docs_for_prompt(candidates)
            out = answer_chain.invoke({"context": context_text, "input": query})
            answer = (out.get("text") or "").strip()

            if not answer or len(answer) < 10 or "Je ne trouve pas" not in answer and relevance_score < RELEVANCE_THRESHOLD:
                answer = "Je ne trouve pas d'informations pertinentes dans ma base de connaissances pour répondre à cette question."

            chat_history.append((query, answer))
            if len(chat_history) > 5:
                chat_history = chat_history[-5:]

            print(f"\n Réponse : {answer}")
            print_sources(candidates)
            print(f"Score de pertinence : {relevance_score:.3f}")

        except KeyboardInterrupt:
            print("\n\nArrêt demandé par l'utilisateur. Au revoir !")
            break
        except Exception as e:
            print(f"\nErreur : {e}")
            print("Veuillez réessayer avec une autre question.")



main()
