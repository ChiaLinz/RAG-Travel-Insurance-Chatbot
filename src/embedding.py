import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from config import EMBEDDING_MODEL

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model=EMBEDDING_MODEL):
    """
    Generate embedding for text using OpenAI.
    """
    res = client.embeddings.create(model=model, input=text)
    return np.array(res.data[0].embedding, dtype="float32")
