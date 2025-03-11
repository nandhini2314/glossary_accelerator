from fastapi import FastAPI, UploadFile, File
import fitz  # PyMuPDF
import re
import openai
import os
from collections import Counter
import nltk
from nltk.corpus import words
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ensure nltk words dataset is available
nltk.download('words')

app = FastAPI()

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file."""
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    text = " ".join([page.get_text("text") for page in doc])
    return text

def get_uncommon_words(text):
    """Extract uncommon words from text."""
    word_list = set(words.words())
    text_words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())  
    word_freq = Counter(text_words)
    uncommon = [word for word in word_freq if word not in word_list and word_freq[word] > 1]
    return uncommon[:7]  

def get_word_meanings(words_list):
    """Use OpenAI's API to get definitions."""
    prompt = f"Provide concise definitions for: {', '.join(words_list)}. Format: word - meaning."
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

@app.post("/generate-glossary/")
async def generate_glossary(pdf: UploadFile = File(...)):
    """API endpoint to generate a glossary from a PDF."""
    text = extract_text_from_pdf(await pdf.read())
    uncommon_words = get_uncommon_words(text)
    glossary = get_word_meanings(uncommon_words) if uncommon_words else "No uncommon words found."
    return {"glossary": glossary}
