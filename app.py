import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from langchain import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

"""
This code sets up a mental health support chatbot using Streamlit and the T5 model from Hugging Face.
It provides users with three types of support: emotional support, mindfulness techniques, and Cognitive Behavioral Therapy (CBT) advice.
Users can interact with the chatbot, view the conversation history, and escalate to a human counselor if needed.
"""

# Define the model path for the T5 model to be used
model_name_or_path = "google/flan-t5-large"

# Load the tokenizer and model for text generation
tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
model = T5ForConditionalGeneration.from_pretrained(
    model_name_or_path,
    low_cpu_mem_usage=True,  # Optimize memory usage for the model
    device_map="cuda:0"      # Set the model to use the first CUDA device for acceleration
)

# Define parameters for text generation
generation_params = {
    "do_sample": True,        # Enable sampling to allow for varied responses
    "temperature": 0.7,       # Control the randomness of predictions
    "top_p": 0.9,             # Apply nucleus sampling to limit the choices of the next token
    "max_new_tokens": 50,     # Specify the maximum number of new tokens to generate
    "repetition_penalty": 1.1  # Penalize repeating tokens in the output
}

# Initialize the text generation pipeline using the specified model and tokenizer
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    **generation_params
)

# Wrap the pipeline in a HuggingFacePipeline for compatibility with LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Define prompts for different types of support
emotional_support_prompt = PromptTemplate(
    input_variables=["conversation_history"],
    template="You are a supportive mental health chatbot. Here is the conversation so far:\n{conversation_history}\nRespond with empathy and understanding."
)

mindfulness_prompt = PromptTemplate(
    input_variables=["conversation_history"],
    template="You are a mindfulness assistant. Here is the conversation so far:\n{conversation_history}\nProvide a brief mindfulness exercise or calming advice."
)

cbt_prompt = PromptTemplate(
    input_variables=["conversation_history"],
    template="You are a CBT (Cognitive Behavioral Therapy) assistant. Here is the conversation so far:\n{conversation_history}\nProvide CBT techniques or advice."
)

# Create chains for each type of support using the defined prompts
emotional_support_chain = LLMChain(llm=llm, prompt=emotional_support_prompt)
mindfulness_chain = LLMChain(llm=llm, prompt=mindfulness_prompt)
cbt_chain = LLMChain(llm=llm, prompt=cbt_prompt)

def escalate_to_human():
    """
    Function to escalate the conversation to a human counselor.
    Returns a message indicating that the escalation has been made.
    """
    return "Your message has been escalated to a human counselor. They will get back to you soon."

def get_chatbot_response(conversation_history, support_type):
    """
    Generate a response from the chatbot based on the user's conversation history and selected support type.
    
    Args:
        conversation_history (str): The current history of the conversation.
        support_type (str): The type of support requested by the user (e.g., emotional support, mindfulness, CBT).
    
    Returns:
        str: The generated response from the chatbot.
    """
    if support_type == 'emotional_support':
        response = emotional_support_chain.run(conversation_history=conversation_history)
    elif support_type == 'mindfulness':
        response = mindfulness_chain.run(conversation_history=conversation_history)
    elif support_type == 'cbt':
        response = cbt_chain.run(conversation_history=conversation_history)
    else:
        response = "Invalid support type. Please choose 'emotional_support', 'mindfulness', or 'cbt'."
    return response

def update_conversation_history(conversation_history, user_message, response, max_history=10):
    """
    Update the conversation history by adding the user's message and the chatbot's response.
    
    Args:
        conversation_history (list): The current conversation history as a list of messages.
        user_message (str): The message from the user.
        response (str): The response from the chatbot.
        max_history (int): Maximum number of conversation pairs to retain.
    
    Returns:
        list: Updated conversation history, retaining only the most recent messages as necessary.
    """
    # Add the new messages to the conversation history
    conversation_history.append(f"You: {user_message}")
    conversation_history.append(f"Chatbot: {response}")

    # Limit the conversation history to the most recent messages
    if len(conversation_history) > max_history * 2:
        conversation_history = conversation_history[-(max_history * 2):]

    return conversation_history

# Main function to run the Streamlit application
def main():
    """
    Main function to set up and run the Streamlit application for the mental health support chatbot.
    This function sets the page layout, handles user input, and displays the conversation.
    """
    st.set_page_config(page_title="Mental Health Support Chatbot", page_icon=":speech_balloon:", layout="wide")
    
    # Custom CSS for styling the application
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to right, #74ebd5, #ACB6E5);
            font-family: 'Open Sans', sans-serif;
            color: #333;
        }
        h1, h2 {
            color: #fff;
            text-shadow: 1px 1px 5px rgba(0, 0, 0, 0.5);
            font-family: 'Montserrat', sans-serif;
        }
        .stTextInput>input {
            background-color: #fff;
            border: 2px solid #007BFF;
            border-radius: 10px;
            padding: 15px;
            font-size: 16px;
        }
        .chat-container {
            background: rgba(255, 255, 255, 0.85);
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .message {
            margin-bottom: 15px;
            padding: 15px;
            border-radius: 20px;
            font-size: 16px;
        }
        .user {
            background: #74ebd5;
            color: #fff;
            text-align: right;
            margin-left: auto;
            max-width: 60%;
            box-shadow: 0 5px 15px rgba(0, 123, 255, 0.3);
        }
        .chatbot {
            background: #444; /* Darker background for chatbot text */
            color: #fff; /* White text for better contrast */
            text-align: left;
            margin-right: auto;
            max-width: 60%;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        }
        .stButton>button {
            background: linear-gradient(90deg, #007BFF, #00FF87);
            color: white;
            border: none;
            border-radius: 15px;
            padding: 10px 20px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #00FF87, #007BFF);
            transform: scale(1.05);
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.title("🌈 Mental Health Support Chatbot")

    # Introduction and instructions for the user
    st.markdown("""
        ### Welcome to your personal support assistant!
        Here, you can receive guidance and emotional care. Choose your preferred type of support below:
        - 🌟 *Emotional Support*: Get empathetic advice and comforting responses.
        - 🌿 *Mindfulness*: Receive mindfulness techniques to help you calm down.
        - 🧠 *CBT*: Learn techniques from Cognitive Behavioral Therapy to manage your thoughts.
        
        📝 Type "I want human support" to escalate to a real counselor.
    """)

    # Dropdown to select the type of support needed
    support_type = st.selectbox(
        "What type of support do you need today?",
        ["emotional_support", "mindfulness", "cbt"]
    )

    # Initialize conversation history if it doesn't already exist in session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Input field for user message
    user_message = st.text_input("Write your message here:")

    col1, col2 = st.columns([1, 1])

    with col1:
        # Handle the send button action
        if st.button("💬 Send"):
            if user_message:
                # Check if the user wants to escalate to human support
                if "I want human support" in user_message:
                    response = escalate_to_human()
                else:
                    # Get a response from the chatbot based on the user's message and selected support type
                    response = get_chatbot_response("\n".join(st.session_state.conversation_history), support_type)

                # Update the conversation history with the new messages
                st.session_state.conversation_history = update_conversation_history(
                    st.session_state.conversation_history, user_message, response
                )

                # Display the conversation history
                st.write("### Conversation")
                with st.container():
                    for message in st.session_state.conversation_history:
                        if "You:" in message:
                            st.markdown(f'<div class="message user">{message}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="message chatbot">{message}</div>', unsafe_allow_html=True)

            else:
                st.warning("Please enter a message to send.")

    with col2:
        # Handle the clear conversation button action
        if st.button("🔄 Clear Conversation"):
            st.session_state.conversation_history = []
            st.success("Conversation cleared. You can start again!")

# Run the application
if __name__ == "__main__":
    main()