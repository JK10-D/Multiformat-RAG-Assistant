# Assistant RAG Multiformat

Assistant conversationnel capable de répondre à des questions
à partir de documents PDF, CSV ou pages web.

## Architecture

URL/PDF/CSV → Loader → Chunker → Embeddings → ChromaDB
↓
Question → Embedding → Recherche vectorielle → Mistral → Réponse

## Stack technique

- **LangChain** — découpage des documents
- **sentence-transformers** — embeddings (all-MiniLM-L6-v2)
- **ChromaDB** — base vectorielle locale
- **Mistral AI** — génération de réponses
- **Streamlit** — interface web

## Installation

```bash
git clone https://github.com/JK10-D/Multiformat-RAG-Assistant
cd rag-assistant
pip install -r requirements.txt
```

Crée un fichier `.env` : MISTRAL_API_KEY=ta-cle-ici

Lance l'application :
```bash
streamlit run app.py
```

## Fonctionnement

1. Colle une URL dans la sidebar et clique "Indexer"
2. Le document est découpé en chunks de 512 tokens
3. Chaque chunk est transformé en vecteur et stocké dans ChromaDB
4. Tes questions retrouvent les 4 chunks les plus pertinents
5. Mistral génère une réponse ancrée dans ces chunks