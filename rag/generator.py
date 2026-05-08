# -*- coding: utf-8 -*-

import os
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
client_llm = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

def build_prompt(question, chunks):
    context = "\n\n---\n\n".join([c["text"] for c in chunks])
    return f"""Tu es un assistant qui répond uniquement à partir des documents fournis.
Si la réponse n'est pas dans les documents, dis-le clairement.

DOCUMENTS:
{context}

QUESTION: {question}"""

def generate(question, chunks, history=[]):
    system_msg = {
        "role": "system",
        "content": build_prompt(question, chunks)
    }
    user_msg = {"role": "user", "content": question}
    messages = [system_msg] + history[-6:] + [user_msg]

    response = client_llm.chat.complete(
        model="mistral-small-latest",
        messages=messages
    )
    return response.choices[0].message.content

