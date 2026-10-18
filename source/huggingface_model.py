from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv


load_dotenv()

messages = [
    SystemMessage("Responda somente com a resposta final, sem mostrar seu raciocínio "
                  "ou as tags <think>. Seja curto e objetivo, em até 140 caracteres."),
    ]

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-3B-Instruct")

model = ChatHuggingFace(llm=llm)

if __name__ == "__main__":
    human_input = input("Digite sua pergunta: ")
    messages.append(HumanMessage(human_input))

    response = model.invoke(messages)

    print(response.content)


# https://pytorch.org/
# Com pensamento:
# https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
# https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
# https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# Direto:
# https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
# https://huggingface.co/Qwen/Qwen2.5-7B-Instruct