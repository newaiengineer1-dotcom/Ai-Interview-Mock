import os
import streamlit as st
from groq import Groq

DEFAULT_MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3-turbo"

def _secret_key():
    try:
        return st.secrets.get("GROQ_API_KEY")
    except Exception:
        return None

def get_client(api_key: str | None = None):
    key = api_key or _secret_key() or os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    return Groq(api_key=key)

def chat(messages, api_key=None, model=DEFAULT_MODEL,
         temperature=0.7, max_tokens=800, json_mode=False):
    client = get_client(api_key)
    if client is None:
        raise RuntimeError("Groq API key not configured.")
    kwargs = dict(model=model, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

def transcribe(audio_path, api_key=None):
    client = get_client(api_key)
    if client is None:
        raise RuntimeError("Groq API key not configured.")
    with open(audio_path, "rb") as f:
        data = f.read()
    resp = client.audio.transcriptions.create(
        file=(os.path.basename(audio_path), data),
        model=WHISPER_MODEL,
        response_format="text",
    )
    return resp if isinstance(resp, str) else getattr(resp, "text", str(resp))
