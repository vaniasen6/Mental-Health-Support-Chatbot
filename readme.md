# Mental Health Support Chatbot

This project is a *Mental Health Support Chatbot* built with Python using the Transformers library and Streamlit for the UI. It provides support in three main categories:
- 🌟 Emotional Support
- 🌿 Mindfulness Exercises
- 🧠 Cognitive Behavioral Therapy (CBT) Techniques

## Features
- Empathetic and comforting responses for emotional support.
- Guided mindfulness exercises.
- CBT techniques to help users manage their thoughts.
- Option to escalate conversations to a human counselor.
- Dynamic conversation history management.
- Simple UI built with Streamlit.

## How It Works
1. *Select Support Type*: Users can choose between "emotional_support", "mindfulness", or "cbt" support.
2. *Chat with the Bot*: Users can input messages and receive responses based on the selected support type.
3. *End the Chat*: Click on clear conversation or I want human support to escalate to a real counselor.

## Tech Stack
- Python
- Hugging Face Transformers for the chatbot's language model (FLAN-T5)
- Streamlit for the user interface
- LangChain for managing conversation history and prompts

## Installation

### Prerequisites
- Python 3.8 or higher
- pip for installing Python packages
- Virtual environment (optional but recommended)
- GPU enabled system (optional but recommended)

### Steps
1. Create and activate a virtual environment:
   bash
   python3 -m venv env
   source env/bin/activate  # On Windows use env\Scripts\activate
    

2. Install the required dependencies:
    bash
    pip install -r requirements.txt
    

3. Run the chatbot:
    bash
    streamlit run app.py
    

4. Access the application in your web browser at http://localhost:8501.

## Usage
- Select the type of support from the dropdown (Emotional Support, Mindfulness, or CBT).
- Enter your message in the input field and press "Send."
- The chatbot will respond with empathetic and supportive advice based on the selected type of support.
- If you want to end the conversation, simply click on clear conversation.
- To escalate the conversation to a human counselor, type I want human support.

## File Structure
- app.py: The main application file for running the chatbot.
- README.md: Project documentation.
- requirements.txt: List of dependencies to run the project.

## Requirements
- Hugging Face Transformers library for the chatbot model.
- Streamlit for building the user interface.
- LangChain for conversation management.

## Future Enhancements
- Add more sophisticated emotional recognition to provide tailored responses.
- Incorporate a larger variety of therapeutic techniques.
- Expand the user interface with animations and additional styling options.

## Disclaimer
This tool is for informational purposes only and does not provide medical advice. Please consult a healthcare provider for a professional diagnosis.