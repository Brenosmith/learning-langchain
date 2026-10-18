import streamlit as st
# from openai_model import model, messages, HumanMessage, AIMessage
from huggingface_model import model, messages, HumanMessage, AIMessage


def openai_chat(user_input, model, messages):
    if "messages" in st.session_state:
        messages = st.session_state.messages
    else:
        st.session_state.messages = messages

    if user_input:
        messages.append(HumanMessage(user_input))
        response = model.invoke(messages)
        messages.append(response)

        if "</think>" in response.content:
            messages.append(AIMessage(response.content.split("</think>")[1]))

    for message in messages:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").write(message.content)

def my_chat_app():
    st.header("Chat com OpenAI", divider=True)
    st.markdown("Converse com o chatGPT")
    user_input = st.chat_input("Digite sua pergunta:")
    openai_chat(user_input, model, messages)

my_chat_app()