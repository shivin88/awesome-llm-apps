🎥 YouTube Video AI Chatbot

This project enables users to interact with YouTube videos by asking questions about their content. The chatbot extracts video transcripts, retrieves relevant sections using embeddings, and generates responses using Google's Gemini AI.

🚀 Features

✅ Extracts and processes YouTube video transcripts✅ Uses Sentence Transformers for embedding-based search✅ Integrates with Google Gemini AI for generating responses✅ Built with Streamlit for an interactive UI✅ Extracts key frames from videos for better context✅ Provides AI-powered summaries and Q&A interactions

🛠️ Installation & Setup

1️⃣ Clone the Repository

git clone https://github.com/your-username/your-repo.git
cd your-repo

2️⃣ Create a Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Set Up API Key Securely

Do NOT hardcode the API key in the script. Instead, create a .env file:

touch .env

Add this to .env:

GEMINI_API_KEY=your_actual_api_key

Then, load it in Python using dotenv:

from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

5️⃣ Run the Application

streamlit run app.py

🔑 Environment Variables

Ensure you have the following variables set:

GEMINI_API_KEY: Your Google Gemini API Key

📜 Usage

Enter a YouTube video URL.

The app fetches and processes the transcript.

Optionally, extract key frames from the video.

Ask questions related to the video's content.

Get AI-powered responses based on the transcript.

📌 Technologies Used

Python (Core development)

Streamlit (Web UI)

YouTube Transcript API (Extracts transcripts)

Sentence Transformers (Embedding-based search)

Google Gemini AI (LLM for generating responses)

OpenCV & PIL (Extracting key frames from videos)

⚠️ Important Security Notice

🔹 Never expose your API key in public repositories.🔹 Add .env to .gitignore to prevent it from being uploaded.

📝 License

This project is open-source and available under the MIT License.

