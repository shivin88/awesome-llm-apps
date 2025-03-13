import tempfile
import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from typing import Tuple, List
import os  # Import the os module
import torch #Import Torch

from langchain.docstore.document import Document
#from langchain.chains.summarize import load_summarize_chain #Switching to Open AI
from transformers import pipeline #Import Transfomers
# Function to initialize the Llama-2 LLM through Hugging Face Inference API
@st.cache_resource #Cache the model to prevent reloading
def initialize_llama_llm():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    generator = pipeline("text-generation", model="facebook/opt-350m", device=device) #CPU. The default is a smaller model
    return generator


# Function to generate response using the Llama-2 LLM
def generate_response(query: str, documents: List[str], llm) -> str:
    try:
        transcript = ' '.join(documents)
        chunk_size = 500  # Adjust chunk size as needed
        chunks = [transcript[i:i+chunk_size] for i in range(0, len(transcript), chunk_size)]

        best_answer = ""
        for chunk in chunks:
            # Revised Prompt:  Directly ask for the answer from the context
            prompt = f"Context: {chunk}\n\nQuestion: {query}\n\nAnswer the question using ONLY the information in the context. If the context does not contain the answer, say 'I cannot answer this question from the given context.'\n\nAnswer:"
            result = llm(prompt, max_length=150, num_return_sequences=1)
            answer = result[0]['generated_text'].strip() # Remove leading/trailing whitespace

            # Improved best_answer logic: Keyword matching
            if any(keyword.lower() in answer.lower() for keyword in query.lower().split()):
                best_answer = answer
                break  # Found a relevant answer, no need to continue
            elif "I cannot answer this question from the given context." not in best_answer and best_answer=="":
                best_answer = answer

        return best_answer
    except Exception as e:
        return f"Error during generation: {e}"

def extract_video_id(video_url: str) -> str:
    """
    Extracts the video ID from different YouTube URL formats.
    """
    patterns = [
        r"youtube\.com/watch\?v=([^&]+)",  # Standard YouTube link
        r"youtube\.com/shorts/([^&]+)",    # YouTube Shorts
        r"youtu\.be/([^&]+)"               # Shortened YouTube link
    ]

    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)

    raise ValueError("Invalid YouTube URL format. Please enter a valid link.")
@st.cache_data #Cache data to prevent re-downloading
def fetch_video_data(video_url: str) -> Tuple[str, str]:
    """
    Fetches the transcript of a YouTube video using its ID.
    """
    try:
        video_id = extract_video_id(video_url)
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US', 'en'])
            transcript_text = " ".join([entry["text"] for entry in transcript])
            return "YouTube Video", transcript_text  # Title set as a placeholder
        except NoTranscriptFound:
            try:
                available_languages = YouTubeTranscriptApi.get_transcript(video_id).find_generated_transcript(['en-US','en']).language_code
                st.warning(f"No English transcript found for this video. Using auto-generated transcript: {available_languages}")
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[available_languages])
                transcript_text = " ".join([entry["text"] for entry in transcript])
                return "YouTube Video", transcript_text
            except Exception as e:
                 st.warning(f"No English transcript found for this video.  This video has transcripts, but not in a supported language.  The following transcripts are available {YouTubeTranscriptApi.list_transcripts(video_id).list_languages()}")
                 return "YouTube Video", "No transcript available for this video."
        except TranscriptsDisabled:
            st.warning("Subtitles are disabled for this video")
            return "YouTube Video", "Subtitles are disabled for this video"


    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return "YouTube Video", "No transcript available for this video."

# Create Streamlit app
st.title("Chat with YouTube Video 📺 (Llama-2 Local)")
st.caption("This app uses a Llama-2 model locally to respond to questions about YouTube videos.")

# Initialize the Llama-2 LLM
llm = initialize_llama_llm()

# Store documents (transcripts) in a list
documents = []

# Get YouTube video URL from the user
video_url = st.text_input("Enter YouTube Video URL")

if video_url:
    try:
        title, transcript = fetch_video_data(video_url)
        if transcript != "No transcript available for this video.":
            documents.append(transcript) #Add the transcript to the list
            st.success(f"Added video '{title}' to the knowledge base!")
        else:
            # Modified message here!
            st.warning(f"No transcript available for video '{title}'.  This video may not have automatically generated subtitles enabled, or transcripts may not be supported. Transcripts are required for this app to function.")
    except ValueError as ve:
        st.error(str(ve))
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    # Ask a question about the video
    prompt = st.text_input("Ask any question about the YouTube Video")

    if prompt:
        if not documents:
            st.warning("Please add a YouTube video transcript first.")
        else:
            try:
                answer = generate_response(prompt, documents, llm)
                st.write(answer)
            except Exception as e:
                st.error(f"Error during response generation: {e}")