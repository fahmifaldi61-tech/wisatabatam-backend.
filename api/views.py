import os
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from rest_framework.views import APIView
from rest_framework.response import Response
from django.core.cache import cache
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()  # baca file .env lokal (kalau ada) dan isi ke environment variables

GROQ_KEY_1 = os.environ.get("GROQ_API_KEY_1")
GROQ_KEY_2 = os.environ.get("GROQ_API_KEY_2")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

groq_client_1 = OpenAI(api_key=GROQ_KEY_1, base_url="https://api.groq.com/openai/v1")
groq_client_2 = OpenAI(api_key=GROQ_KEY_2, base_url="https://api.groq.com/openai/v1")
gemini_client = genai.Client(api_key=GEMINI_KEY)

try:
    embeddings = FastEmbedEmbeddings()
    vectorstore = Chroma(persist_directory="./chroma_db_local", embedding_function=embeddings)
except Exception:
    vectorstore = None


def ask_groq(client, prompt):
    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
    )
    return completion.choices[0].message.content


def ask_gemini(prompt):
    interaction = gemini_client.interactions.create(
        model="gemini-flash-latest",
        input=prompt,
    )
    return interaction.output_text


def ask_ai_with_fallback(prompt):
    attempts = [
        ("Groq (key 1)", lambda: ask_groq(groq_client_1, prompt)),
        ("Groq (key 2)", lambda: ask_groq(groq_client_2, prompt)),
        ("Gemini", lambda: ask_gemini(prompt)),
    ]
    errors = []
    for name, fn in attempts:
        try:
            result = fn()
            return result, name, errors
        except Exception as e:
            errors.append(f"{name}: {str(e)}")
            continue
    raise Exception(" | ".join(errors))


LANGUAGE_INSTRUCTIONS = {
    "id": (
        "Anda adalah asisten virtual pariwisata Kota Batam. "
        "WAJIB menjawab dalam Bahasa Indonesia yang ramah dan singkat, "
        "meskipun konteks data atau pertanyaan tercampur bahasa lain."
    ),
    "en": (
        "You are a virtual tourism assistant for Batam City. "
        "You MUST answer in friendly, concise English, "
        "even if the data context or question is in another language."
    ),
}

# --- KONFIGURASI CACHE PERCAKAPAN (CONTEXT MEMORY) ---
CACHE_TIMEOUT = 60 * 60 * 24   # 24 jam, dalam detik
MAX_HISTORY_TURNS = 15         # simpan 15 pertukaran terakhir biar AI makin \"inget banyak\"


class ChatAPIView(APIView):
    def post(self, request):
        user_input = request.data.get('message', '')
        language = request.data.get('language', 'id')
        session_id = request.data.get('session_id') or str(uuid.uuid4())

        if language not in LANGUAGE_INSTRUCTIONS:
            language = 'id'
        if not user_input:
            return Response({'error': 'Pesan kosong'}, status=400)

        # --- Ambil riwayat percakapan dari cache (kalau ada & belum expired) ---
        cache_key = f"chat_history_{session_id}"
        history = cache.get(cache_key, [])

        context = ""
        sources = []
        if vectorstore:
            try:
                docs = vectorstore.similarity_search(user_input, k=3)
                context = "\n".join([doc.page_content for doc in docs])

                seen = set()
                for doc in docs:
                    src = doc.metadata.get("source") if doc.metadata else None
                    if src and src not in seen:
                        seen.add(src)
                        sources.append(src)
            except Exception:
                pass

        # --- Susun riwayat percakapan jadi teks buat dimasukin ke prompt ---
        history_text = ""
        if history:
            lines = []
            for turn in history[-MAX_HISTORY_TURNS:]:
                lines.append(f"User: {turn['user']}")
                lines.append(f"AI: {turn['ai']}")
            history_text = "\n".join(lines)

        instruction = LANGUAGE_INSTRUCTIONS[language]
        prompt = (
            f"{instruction}\n\n"
            f"Riwayat percakapan sebelumnya (gunakan sebagai konteks bila relevan, "
            f"abaikan kalau tidak nyambung dengan pertanyaan sekarang):\n{history_text}\n\n"
            f"Konteks Data / Data Context:\n{context}\n\n"
            f"Pertanyaan User / User Question: {user_input}\n"
        )

        try:
            result, used_provider, errors = ask_ai_with_fallback(prompt)

            # --- Simpan percakapan ini ke cache, reset TTL jadi 24 jam lagi ---
            history.append({'user': user_input, 'ai': result})
            history = history[-MAX_HISTORY_TURNS:]
            cache.set(cache_key, history, CACHE_TIMEOUT)

            return Response({
                'response': result,
                'provider_used': used_provider,
                'language': language,
                'sources': sources,
                'session_id': session_id,
            })
        except Exception as e:
            return Response({'error': f"Semua provider gagal: {str(e)}"}, status=500)
