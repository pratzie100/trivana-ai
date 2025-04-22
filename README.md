# 🎭 TRIVANA AI — Your Triple-Domain Culture Companion

**TRIVANA AI** is a smart conversational assistant powered by Groq + LangChain, specialized in answering questions across **Movies**, **Music**, and **Sports** domains — while keeping the conversation genre-focused and context-aware.

### 🚀 Features

- 🎬 **Movies**: Recommendations, cast info, trivia, release updates.
- 🎵 **Music**: Songs, albums, lyrics, artist info, and mood-based suggestions.
- 🏆 **Sports**: Match scores, player stats, history, rules, and fixtures.
- 🧠 **Conversational memory**: Retains context for up to a configurable number of interactions.
- 🧾 **Error Handling**: Manages Groq API rate limits and off-topic questions gracefully.
- 🌈 **Modern UI**: Dark-themed, responsive Streamlit interface with custom styling.
- 🖥️ **Model Selection**: Allows users to choose from various Groq models for tailored responses.

---

## 🛠️ Built With

| Tech/Library      | Usage                              |
|-------------------|-------------------------------------|
| `Streamlit`       | Frontend Web App Interface 🖥️        |
| `Groq API`        | LLMs powering chat responses ⚡     |
| `LangChain`       | Memory & Runnables for history 🧠    |
| `Python-dotenv`   | Secure API key handling 🔐          |
| `HTML/CSS`        | Custom theming via markdown 🎨      |

### ⚙️ Setup Instructions

    ```bash
    git clone https://github.com/pratzie100/trivana-ai.git 
    cd trivana-ai 

THEN IN YOUR LOCAL ENVIRONMENT

    ```bash
    pip install -r requirements.txt 
    streamlit run app.py


Make sure to create a `.env` file in the root directory with your Groq API key.


## 🧪 LIVE DEMO

Visit [this link](https://trivana-ai-pratyush-kargeti.streamlit.app/) to try out the application in your browser.