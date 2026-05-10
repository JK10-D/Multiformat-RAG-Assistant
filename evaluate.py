import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from rag.loader import load_document
from rag.chunker import create_chunks
from rag.embedder import index_chunks, search
from rag.generator import generate
from mistralai import Mistral

client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Étape 1 — Charge et indexe
print("Chargement du document...")
texte = load_document("https://fr.wikipedia.org/wiki/Intelligence_artificielle")
chunks = create_chunks(texte, "wikipedia_IA")
index_chunks(chunks)

# Étape 2 — Jeu de test
questions = [
    "Qui a introduit le terme intelligence artificielle ?",
    "Qu'est-ce que l'intelligence artificielle étroite ?",
    "Quelle est la différence entre IA faible et IA forte ?",
    "Qu'est-ce que le machine learning ?",
    "Qui est Alan Turing et quel est son lien avec l'IA ?",
]

ground_truths = [
    "Le terme intelligence artificielle a été introduit par John McCarthy en 1956.",
    "L'intelligence artificielle étroite est conçue pour effectuer une tâche précise.",
    "L'IA faible est spécialisée dans une tâche précise tandis que l'IA forte peut fonctionner sur une large gamme de tâches.",
    "Le machine learning permet à un système d'apprendre à partir de données sans être explicitement programmé.",
    "Alan Turing est un mathématicien considéré comme l'un des pères de l'informatique et de l'IA.",
]

# Étape 3 — Génère les réponses
print("\nGénération des réponses...")
answers = []
contexts = []
for question in questions:
    resultats = search(question, top_k=4)
    reponse = generate(question, resultats)
    answers.append(reponse)
    contexts.append([r["text"] for r in resultats])
    print(f"Q: {question[:50]}... → OK")

# Étape 4 — Évaluation manuelle avec Mistral
def evaluate_faithfulness(question, answer, context_list):
    context = "\n".join(context_list)
    prompt = f"""Tu es un évaluateur. Réponds uniquement par un score entre 0 et 1.
La réponse est-elle entièrement basée sur le contexte fourni ? (1 = oui totalement, 0 = non du tout)

Contexte: {context[:1000]}
Réponse: {answer}

Score (uniquement un nombre entre 0 et 1):"""
    
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0), 1)
    except:
        return 0.5

def evaluate_relevancy(question, answer):
    prompt = f"""Tu es un évaluateur. Réponds uniquement par un score entre 0 et 1.
La réponse répond-elle bien à la question ? (1 = parfaitement, 0 = pas du tout)

Question: {question}
Réponse: {answer}

Score (uniquement un nombre entre 0 et 1):"""
    
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        score = float(response.choices[0].message.content.strip())
        return min(max(score, 0), 1)
    except:
        return 0.5

# Étape 5 — Calcule les scores
print("\nÉvaluation en cours...")
faithfulness_scores = []
relevancy_scores = []

for i, (q, a, c) in enumerate(zip(questions, answers, contexts)):
    f_score = evaluate_faithfulness(q, a, c)
    r_score = evaluate_relevancy(q, a)
    faithfulness_scores.append(f_score)
    relevancy_scores.append(r_score)
    print(f"Q{i+1}: Faithfulness={f_score:.2f} | Relevancy={r_score:.2f}")

avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
avg_relevancy = sum(relevancy_scores) / len(relevancy_scores)

print("\n=== RÉSULTATS FINAUX ===")
print(f"Faithfulness moyen     : {avg_faithfulness:.3f}")
print(f"Answer Relevancy moyen : {avg_relevancy:.3f}")
print("\nInterprétation :")
print("- Faithfulness > 0.8 = réponses bien ancrées dans les documents")
print("- Answer Relevancy > 0.8 = réponses bien adaptées aux questions")