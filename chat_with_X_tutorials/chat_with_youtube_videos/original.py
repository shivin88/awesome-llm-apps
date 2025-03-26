import asyncio
import sys
import re
import numpy as np
from typing import List, Tuple
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# 🛠 MacOS-specific fix for AsyncIO event loop issues
if sys.platform == "darwin":  
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception as e:
        print(f"[ERROR] AsyncIO fix failed: {e}")

# 🔑 Set Google Gemini API Key
GEMINI_API_KEY = "AIzaSyA8SiOVTffZhAIk-13m_7jBo55FZsZg7c0"
genai.configure(api_key=GEMINI_API_KEY)

# 🔍 Initialize Sentence Transformer model
def initialize_embedding_model():
    print("[DEBUG] Initializing Sentence Transformer model...")
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
            print(f"[DEBUG] Extracted Video ID: {match.group(1)}")
            return match.group(1)
    raise ValueError("Invalid YouTube URL format.")

# 📝 Fetch Video Transcript
def fetch_video_data(video_url: str) -> Tuple[str, str]:
    try:
        video_id = extract_video_id(video_url)
        print("[DEBUG] Fetching transcript...")
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US', 'en'])
            print("[DEBUG] Fetched manually provided transcript.")
        except NoTranscriptFound:
            try:
                available_transcript = YouTubeTranscriptApi.list_transcripts(video_id).find_generated_transcript(['en-US', 'en'])
                transcript = available_transcript.fetch()
                print("[DEBUG] Using auto-generated transcript.")
            except Exception:
                print("[ERROR] No transcript available.")
                return "YouTube Video", "No transcript available."
        except TranscriptsDisabled:
            print("[ERROR] Subtitles are disabled for this video.")
            return "YouTube Video", "Subtitles are disabled for this video."
        
        transcript_text = " ".join(entry["text"] for entry in transcript)
        print(f"[DEBUG] Transcript Length: {len(transcript_text)} characters")
        return "YouTube Video", transcript_text
    except Exception as e:
        print(f"[ERROR] Error fetching transcript: {e}")
        return "YouTube Video", f"Error fetching transcript: {e}"

# 📜 Split Transcript into Chunks
def split_transcript_into_chunks(transcript: str, chunk_size: int = 1000) -> List[str]:
    words = transcript.split()
    if not words:
        print("[ERROR] Transcript is empty after fetching.")
        return []
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    print(f"[DEBUG] Transcript split into {len(chunks)} chunks.")
    return chunks

# 🔎 Retrieve Relevant Chunks Using Embeddings
def retrieve_relevant_chunks(query: str, chunks: List[str], embedding_model, top_k: int = 5) -> List[str]:
    if not chunks:
        print("[ERROR] No chunks available for retrieval.")
        return ["No relevant information found."]
    
    print("[DEBUG] Encoding query and transcript chunks...")
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0].cpu().numpy()
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices if similarities[i] > 0.2]  # Lowered threshold
    
    print("[DEBUG] Top Matching Chunks:")
    for i, idx in enumerate(top_indices):
        print(f"Chunk {idx + 1}: Similarity = {similarities[idx]:.4f} - {chunks[idx][:100]}...")
    
    return relevant_chunks if relevant_chunks else ["No relevant information found."]

# 🤖 Generate Response Using Gemini AI
def generate_response(query: str, relevant_chunks: List[str], llm) -> str:
    try:
        transcript_text = "\n".join(relevant_chunks) if relevant_chunks else "No relevant information found."
        print(f"\n[DEBUG] Transcript Snippets Passed to LLM:\n{transcript_text[:500]}")

        prompt = f"""
        You are an AI assistant answering questions about a YouTube video. Use the transcript below to generate a response.
        If the transcript lacks details, supplement your answer with general knowledge, but make it clear when you're doing so.

        *Transcript Snippets:*  
        {transcript_text}

        *User Question:* {query}

        *Final Answer:*  
        """

        print("[DEBUG] Sending prompt to LLM...")
        response = llm.generate_content(prompt)  
        print("[DEBUG] LLM response received.")

        return response.text if response else "No response generated."
    
    except Exception as e:
        print(f"[ERROR] Error during response generation: {e}")
        return f"Error during generation: {e}"

# 🚀 Main Function to Run the Chatbot
def main():
    print("Chat with YouTube Video 📺 (Gemini API)")
    
    llm = genai.GenerativeModel("gemini-1.5-flash")
    embedding_model = initialize_embedding_model()
    
    video_url = input("Enter YouTube Video URL: ").strip()
    if not video_url:
        print("[ERROR] Invalid input. Please enter a valid URL.")
        return
    
    title, transcript = fetch_video_data(video_url)
    if transcript in ["No transcript available.", "Subtitles are disabled for this video."]:
        print(f"[ERROR] {transcript}")
        return
    
    print(f"[DEBUG] Added video '{title}' to knowledge base!")
    
    prompt = input("Ask a question about the video: ").strip()
    if prompt:
        chunks = split_transcript_into_chunks(transcript)
        relevant_chunks = retrieve_relevant_chunks(prompt, chunks, embedding_model)
        answer = generate_response(prompt, relevant_chunks, llm)  
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
