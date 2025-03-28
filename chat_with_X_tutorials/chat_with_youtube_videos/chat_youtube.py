import streamlit as st
import asyncio
import sys
import re
import numpy as np
from typing import List, Tuple
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# ✅ MacOS-specific fix for AsyncIO event loop issues
if sys.platform == "darwin":
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception as e:
        print(f"[ERROR] AsyncIO fix failed: {e}")

# 🔑 Set Google Gemini API Key (Replace with your actual API key)
GEMINI_API_KEY = "add api key"
genai.configure(api_key=GEMINI_API_KEY)

# 🔍 Initialize Sentence Transformer model
@st.cache_resource
def initialize_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# 🎥 Extract YouTube Video ID
def extract_video_id(video_url: str) -> str:
    patterns = [
        r"youtube\.com/watch\?v=([^&]+)",
        r"youtube\.com/shorts/([^&]+)",
        r"youtu\.be/([^&]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL format.")

# 📝 Fetch Video Transcript
def fetch_video_data(video_url: str) -> Tuple[str, str]:
    try:
        video_id = extract_video_id(video_url)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US', 'en'])
        except NoTranscriptFound:
            try:
                available_transcript = YouTubeTranscriptApi.list_transcripts(video_id).find_generated_transcript(['en-US', 'en'])
                transcript = available_transcript.fetch()
            except Exception:
                return "YouTube Video", "No transcript available."
        except TranscriptsDisabled:
            return "YouTube Video", "Subtitles are disabled for this video."

        transcript_text = " ".join(entry["text"] for entry in transcript)
        return "YouTube Video", transcript_text
    except Exception as e:
        return "YouTube Video", f"Error fetching transcript: {e}"

# 📜 Split Transcript into Chunks
def split_transcript_into_chunks(transcript: str, chunk_size: int = 1000) -> List[str]:
    words = transcript.split()
    if not words:
        return []
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# 🔎 Retrieve Relevant Chunks Using Embeddings
def retrieve_relevant_chunks(query: str, chunks: List[str], embedding_model, top_k: int = 5) -> List[str]:
    if not chunks:
        return ["No relevant information found."]

    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0].cpu().numpy()

    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices if similarities[i] > 0.2]  # Lowered threshold
    
    return relevant_chunks if relevant_chunks else ["No relevant information found."]

# 🤖 Generate Response Using Gemini AI
def generate_response(query: str, relevant_chunks: List[str], llm) -> str:
    try:
        transcript_text = "\n".join(relevant_chunks) if relevant_chunks else "No relevant information found."
        
        prompt = f"""
        You are an AI assistant answering questions about a YouTube video. Use the transcript below to generate a response.
        If the transcript lacks details, supplement your answer with general knowledge, but make it clear when you're doing so.

        *Transcript Snippets:*  
        {transcript_text}

        *User Question:* {query}

        *Final Answer:*  
        """

        response = llm.generate_content(prompt)  
        return response.text if response else "No response generated."
    
    except Exception as e:
        return f"Error during generation: {e}"

# 🚀 Streamlit UI
def main():
    st.set_page_config(page_title="YouTube Video Chatbot", layout="wide")
    st.title("📺 Chat with YouTube Videos using AI 🎙️")

    llm = genai.GenerativeModel("gemini-1.5-flash")
    embedding_model = initialize_embedding_model()

    video_url = st.text_input("Enter YouTube Video URL", "")
    if video_url:
        title, transcript = fetch_video_data(video_url)
        if transcript in ["No transcript available.", "Subtitles are disabled for this video."]:
            st.error(transcript)
            return

        st.success(f"✅ Video '{title}' added to knowledge base!")
        query = st.text_input("Ask a question about the video", "")

        if query:
            with st.spinner("🔍 Searching for relevant parts..."):
                chunks = split_transcript_into_chunks(transcript)
                relevant_chunks = retrieve_relevant_chunks(query, chunks, embedding_model)

            with st.spinner("🤖 Generating AI-powered response..."):
                answer = generate_response(query, relevant_chunks, llm)

            st.subheader("💡 AI Response:")
            st.write(answer)

if __name__ == "__main__":
    main()
