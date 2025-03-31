import streamlit as st

# Must be the first Streamlit command
st.set_page_config(
    page_title="YouTube Video AI Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import other modules
import asyncio
import sys
import re
import os
import numpy as np
from typing import List, Tuple
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import cv2
import tempfile
from PIL import Image

# Rest of your imports and code...

# ✅ Fix AsyncIO issues on MacOS
if sys.platform == "darwin":
    try:
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception as e:
        print(f"[ERROR] AsyncIO fix failed: {e}")

# 🔑 Load API Key from Environment Variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    st.error("⚠️ Missing GEMINI API Key. Please set `GEMINI_API_KEY` in your environment.")
    sys.exit(1)
genai.configure(api_key=GEMINI_API_KEY)

# Custom CSS for improved UI
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stTextInput>div>div>input {
            border-radius: 20px;
            padding: 10px 15px;
        }
        .stButton>button {
            border-radius: 20px;
            padding: 10px 24px;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .summary-box {
            border-radius: 10px;
            padding: 20px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .response-box {
            border-radius: 10px;
            padding: 20px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 20px;
        }
        .frame-container {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .header {
            color: #2c3e50;
        }
        .sidebar .sidebar-content {
            background-color: #2c3e50;
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)

# [Rest of your functions remain exactly the same...]
# 🎥 Extract YouTube Video ID
def extract_video_id(video_url: str) -> str:
    match = re.search(r"(?:v=|be/|shorts/)([A-Za-z0-9_-]{11})", video_url)
    return match.group(1) if match else None

# 🎬 Download YouTube Video using yt-dlp
def download_video(video_url: str) -> str:
    video_id = extract_video_id(video_url)
    if not video_id:
        raise ValueError("❌ Invalid YouTube URL.")

    output_path = f"downloads/{video_id}.mp4"
    os.makedirs("downloads", exist_ok=True)

    ydl_opts = {
        "format": "best",
        "outtmpl": output_path,
        "quiet": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        return output_path
    except Exception as e:
        raise RuntimeError(f"❌ Error downloading video: {e}")

# 📝 Fetch YouTube Video Transcript
def fetch_video_transcript(video_url: str) -> str:
    video_id = extract_video_id(video_url)
    if not video_id:
        return "Invalid YouTube URL"

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    except NoTranscriptFound:
        return "No transcript available."
    except TranscriptsDisabled:
        return "Subtitles are disabled for this video."
    
    return " ".join(entry["text"] for entry in transcript)

# 🖼️ Extract Key Frames from Video
def extract_key_frames(video_path: str, num_frames: int = 3) -> List[Image.Image]:
    if not os.path.exists(video_path):
        st.error("❌ Video file not found.")
        return []
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [int(total_frames * (i / (num_frames + 1))) for i in range(1, num_frames + 1)]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames

# 📝 Split Transcript into Chunks
def split_transcript_into_chunks(transcript: str, chunk_size: int = 1000) -> List[str]:
    words = transcript.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)] if words else []

# 🔎 Retrieve Relevant Chunks Using Embeddings
@st.cache_resource
def initialize_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_relevant_chunks(query: str, chunks: List[str], embedding_model, top_k: int = 5) -> List[str]:
    if not chunks:
        return ["No relevant information found."]
    
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    chunk_embeddings = embedding_model.encode(chunks, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0].cpu().numpy()
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices if similarities[i] > 0.2]

# 🤖 Generate AI Response
def generate_response(query: str, relevant_chunks: List[str], llm, images: List[Image.Image] = None) -> str:
    try:
        transcript_text = "\n".join(relevant_chunks) if relevant_chunks else "No relevant information found."
        
        prompt = f"""
        You are an AI assistant answering questions about a YouTube video. Use the transcript below to generate a response.
        
        *Transcript Snippets:*  
        {transcript_text}
        
        *User Question:* {query}
        
        *Final Answer:*  
        """

        if images:
            response = llm.generate_content([prompt] + images)
        else:
            response = llm.generate_content(prompt)

        return response.text if response else "No response generated."
    except Exception as e:
        return f"Error during generation: {e}"

# 🚀 Streamlit UI
def main():
    # Apply custom CSS
    local_css()
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=100)
        st.title("YouTube Video AI")
        st.markdown("""
        **Chat with any YouTube video** by extracting its transcript and key frames.
        Ask questions, get summaries, and analyze video content with AI.
        """)
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("1. Enter YouTube URL")
        st.markdown("2. Click 'Show Video Frames' if needed")
        st.markdown("3. Ask questions or get a summary")
        st.markdown("4. Download Q&A session if needed")
        st.markdown("---")
        st.markdown("Made with using Streamlit & Gemini AI")
    
    # Main content
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎬 YouTube Video AI Assistant")
        st.markdown("Extract insights from any YouTube video with AI-powered analysis")
    
    # Initialize all session state variables
    if "video_images" not in st.session_state:
        st.session_state.video_images = []
    if "show_images" not in st.session_state:
        st.session_state.show_images = False
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []
    if "processed_queries" not in st.session_state:
        st.session_state.processed_queries = set()

    llm = genai.GenerativeModel("gemini-1.5-flash")
    embedding_model = initialize_embedding_model()

    # Video URL input
    with st.container():
        video_url = st.text_input("**Enter YouTube Video URL**", "", 
                                placeholder="https://www.youtube.com/watch?v=...")
    
    if video_url:
        with st.spinner("🔍 Processing video..."):
            transcript = fetch_video_transcript(video_url)
            if "No transcript" in transcript or "Subtitles are disabled" in transcript:
                st.error(transcript)
            else:
                st.success("✅ Video transcript loaded successfully!")
                
                # Action buttons row
                col1, col2, col3 = st.columns([1,1,2])
                with col1:
                    if st.button("🖼️ Show Video Frames", type="primary"):
                        st.session_state.show_images = not st.session_state.show_images
                with col2:
                    summarize_btn = st.button("📝 Get Summary", type="secondary")
                with col3:
                    pass
                
                # Extract key frames if requested
                if st.session_state.show_images and not st.session_state.video_images:
                    with st.spinner("🖼️ Extracting key frames..."):
                        try:
                            video_file = download_video(video_url)
                            st.session_state.video_images = extract_key_frames(video_file)
                        except Exception as e:
                            st.error(f"Failed to extract frames: {str(e)}")
                
                # Show images if available and requested
                if st.session_state.show_images and st.session_state.video_images:
                    st.subheader("🎬 Key Frames from Video")
                    cols = st.columns(len(st.session_state.video_images))
                    for idx, (col, img) in enumerate(zip(cols, st.session_state.video_images)):
                        with col:
                            with st.container():
                                st.markdown(f'<div class="frame-container">', unsafe_allow_html=True)
                                st.image(img, caption=f"Frame {idx+1}", use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                
                # Summary section
                if summarize_btn:
                    query = "Summarize the video"
                    if query not in st.session_state.processed_queries:
                        with st.spinner("📖 Generating summary..."):
                            answer = generate_response(
                                query, 
                                [transcript], 
                                llm, 
                                st.session_state.video_images if st.session_state.show_images else None
                            )
                        with st.container():
                            st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                            st.subheader("📌 Video Summary")
                            st.write(answer)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.session_state.qa_history.append(f"Q: {query}\nA: {answer}\n\n")
                            st.session_state.processed_queries.add(query)
                
                # Question input
                with st.container():
                    st.subheader("💬 Ask About the Video")
                    query = st.text_input("**Your question**", "", 
                                        placeholder="What cars are featured in this video?",
                                        key="question_input")
                    
                    if query and query not in st.session_state.processed_queries:
                        with st.spinner("🧠 Thinking..."):
                            chunks = split_transcript_into_chunks(transcript)
                            relevant_chunks = retrieve_relevant_chunks(query, chunks, embedding_model)
                            answer = generate_response(
                                query, 
                                relevant_chunks, 
                                llm, 
                                st.session_state.video_images if st.session_state.show_images else None
                            )
                        
                        with st.container():
                            st.markdown('<div class="response-box">', unsafe_allow_html=True)
                            st.subheader("💡 AI Response")
                            st.write(answer)
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.session_state.qa_history.append(f"Q: {query}\nA: {answer}\n\n")
                            st.session_state.processed_queries.add(query)
                
                # Download Q&A
                if st.session_state.qa_history:
                    with st.container():
                        st.markdown("---")
                        st.subheader("📥 Download Session")
                        if st.button("💾 Download Q&A History"):
                            # Remove duplicates while maintaining order
                            seen = set()
                            unique_qa_history = []
                            for qa in st.session_state.qa_history:
                                if qa not in seen:
                                    seen.add(qa)
                                    unique_qa_history.append(qa)
                            
                            st.download_button(
                                label="⬇️ Download Q&A as TXT",
                                data="".join(unique_qa_history),
                                file_name="YouTube_QA_Session.txt",
                                mime="text/plain",
                                key="download_qa"
                            )

if __name__ == "__main__":
    main()