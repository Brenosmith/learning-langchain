from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv


load_dotenv()

messages = [
    SystemMessage("Responda as perguntas de forma curta e objetiva, em até 140 caracteres."),
    ]

model = ChatOpenAI()

if __name__ == "__main__":
    human_input = input("Digite sua pergunta: ")
    messages.append(HumanMessage(human_input))

    response = model.invoke(messages)

    print(response.content)
