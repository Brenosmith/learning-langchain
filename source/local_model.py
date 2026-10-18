from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import torch
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface.chat_models import ChatHuggingFace


# MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

# quantization_config = BitsAndBytesConfig(
#     load_in_8bit=True,
# )

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

causal_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    # quantization_config=quantization_config,
    dtype=torch.float16,
)

text_generation_pipeline = pipeline(
    "text-generation",
    model=causal_model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(
        content="Responda somente com a resposta final, sem mostrar seu raciocínio "
                  "ou as tags <think>. Seja curto e objetivo, em até 140 caracteres."),
]

if __name__ == "__main__":
    human_input = input("Digite sua pergunta: ")
    messages.append(HumanMessage(content=human_input))

    response = model.invoke(messages)

    print(response.content)

    print("Memória alocada (GB):", torch.cuda.memory_allocated() / 1024**3)
    print("Memória reservada (GB):", torch.cuda.memory_reserved() / 1024**3)
