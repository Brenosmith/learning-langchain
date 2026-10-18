from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv


load_dotenv()

messages = [
    SystemMessage("Responda as perguntas de forma curta e objetiva, em até 140 caracteres."),
    ]

llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")

model = ChatHuggingFace(llm=llm)

if __name__ == "__main__":
    human_input = input("Digite sua pergunta: ")
    messages.append(HumanMessage(human_input))

    response = model.invoke(messages)

    print(response.content)


# https://pytorch.org/
# https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
