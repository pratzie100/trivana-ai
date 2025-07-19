import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please configure it in environment variables.")
    st.stop()

# Session-level memory store
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Define polished genre prompts (unchanged)
MOVIES_PROMPT = """
🎬 **Welcome to the Movie Zone!**
You can ask me anything related to:
- Movie recommendations (by genre, actor, or mood)
- Latest releases and upcoming movies
- Bollywood, Hollywood, Tollywood — anything cinema!
- Reviews, plots, cast info, or director trivia
❗ *Please ask only movie-related questions.*
If you ask about music, sports, or anything else, I’ll kindly ask you to stick to movies. 😊
"""

MUSIC_PROMPT = """
🎵 **Welcome to the Music Lounge!**
Ask me about:
- Songs, albums, or artists across genres (pop, hip-hop, classical, etc.)
- Lyrics, music video links, or streaming platforms
- Recommendations based on mood or artists
❗ *Please ask only music-related questions.*
If you ask about movies, sports, or other topics, I’ll gently guide you back to music. 🎷
"""

SPORTS_PROMPT = """
🏅 **Welcome to the Sports Arena!**
I can help you with:
- Latest match scores, results, and upcoming fixtures
- Player stats, team rankings, and tournament news
- Sports history and trivia
- Game rules, strategies, or how-tos
❗ *Please keep your questions sports-related.*
If you ask about movies, music, or unrelated topics, I’ll remind you to stay in the game! 🏆
"""

GENRE_PROMPTS = {
    "Movies": MOVIES_PROMPT,
    "Music": MUSIC_PROMPT,
    "Sports": SPORTS_PROMPT
}

def main():
    st.set_page_config(page_title="TRIVANA AI", layout="wide")

    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to bottom right, #1a1a2e, #2e2e3e);
            color: #e0e0e0;
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .stChatMessage {
            margin-bottom: 15px;
        }
        .stChatMessage[data-testid="chatAvatarIcon-user"] {
            background: #4a90e2 !important;
            border-radius: 8px;
            padding: 10px;
            max-width: 70%;
            margin-left: auto;
            color: #ffffff;
        }
        .stChatMessage[data-testid="chatAvatarIcon-assistant"] {
            background: #555566 !important;
            border-radius: 8px;
            padding: 10px;
            max-width: 70%;
            margin-right: auto;
            color: #e0e0e0;
        }
        .header-container {
            text-align: center;
            padding: 20px 0;
            margin-bottom: 20px;
        }
        .title {
            font-size: 48px;
            font-weight: bold;
            color: #ffffff;
            margin: 0;
        }
        .tagline {
            font-size: 16px;
            color: #b0b0b0;
            margin-top: 8px;
        }
        .footer {
            margin-top: auto;
            text-align: center;
            padding: 15px 0;
            font-size: 14px;
            color: #b0b0b0;
            border-top: 1px solid #3a3a4a;
            width: 100%;
        }
        .footer a {
            color: #4FC3F7;
            text-decoration: none;
            margin: 0 5px;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header with big title and small tagline
    st.markdown("""
    <div class='header-container'>
        <h1 class='title'>TRIVANA AI</h1>
        <div class='tagline'>Your triple-domain culture companion.</div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar settings (unchanged)
    st.sidebar.title("Settings")
    model = st.sidebar.selectbox("Choose a model", ["gemma2-9b-it", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    genre = st.sidebar.selectbox("Choose a genre", ["Music", "Sports", "Movies"])
    conversational_memory_length = st.sidebar.slider("Conversational memory length:", 1, 10, value=5)

    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    system_prompt = GENRE_PROMPTS.get(genre)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])

    chain = prompt | groq_chat
    runnable_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history"
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message["human"])
        with st.chat_message("assistant"):
            st.markdown(message["AI"])

    user_question = st.chat_input("Ask anything from choosen genre :)")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        try:
            response = runnable_with_history.invoke(
                {"input": user_question},
                config={"configurable": {"session_id": "user_session"}}
            )

            if genre == "Movies" and any(x in user_question.lower() for x in ["song", "album", "lyrics", "player", "score", "match"]):
                content = "❌ Your question seems to be outside the selected genre (Movies). Please ask only about movies."
            elif genre == "Music" and any(x in user_question.lower() for x in ["movie", "film", "score", "match", "team"]):
                content = "❌ Your question seems to be outside the selected genre (Music). Please stick to music-related questions."
            elif genre == "Sports" and any(x in user_question.lower() for x in ["song", "album", "movie", "film"]):
                content = "❌ Your question seems to be outside the selected genre (Sports). Please ask only sports-related stuff."
            else:
                content = response.content

            with st.chat_message("assistant"):
                st.markdown(content)

            st.session_state.chat_history.append({"human": user_question, "AI": content})

            session_history = get_session_history("user_session")
            if len(session_history.messages) > conversational_memory_length * 2:
                session_history.messages = session_history.messages[-conversational_memory_length * 2:]

        except Exception as e:
            if "429" in str(e):
                st.error("Rate limit exceeded. Please try again later or check your Groq dashboard.")
            else:
                st.error(f"Error: {str(e)}")

    # Footer
    st.markdown("""
    <div class='footer'>
        Made with ❤️ by Pratyush Kargeti |
        <a href="https://github.com/pratzie100" target="_blank">Github</a> |
        <a href="https://www.linkedin.com/in/pratyush-kargeti-576270285" target="_blank">LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


