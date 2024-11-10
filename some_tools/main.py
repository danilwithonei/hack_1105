import gradio as gr
from ollama import Client
from settings import host
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import re
import os
import shutil
from pprint import pprint

patg_to_upload = "upload_data"
path_to_save_prompts = "results"

def extract_integers_from_string(s):
    """
    Extracts all integer numbers from a given string.

    Parameters:
    s (str): The input string from which to extract integers.

    Returns:
    List[int]: A list of integers extracted from the string.
    """
    # Use regular expression to find all integer numbers (including negative numbers)
    numbers = re.findall(r"-?\d+", s)
    # Convert the extracted strings to integers
    integers = [int(num) for num in numbers]
    return integers


client = Client(host=host)
model = SentenceTransformer("intfloat/multilingual-e5-large")

milvus_client = MilvusClient(uri="./milvus_demo.db")
collection_name = "my_rag_collection"


def emb_text(text):
    embeddings = model.encode(text)
    return embeddings


def test_inference(prompt, context):
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            emb_text(prompt)
        ],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=5,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text", "file_name", "page"],  # Return the text field
    )
    search_context = ""
    for i, res in enumerate(search_res[0]):
        search_context += f'текст № {i}:\n {res["entity"]["text"]}\n'

    prompt2 = f"""
## Задание
Ты специалист по работе с данными, ответь на вопрос используя контекст. Дай краткий ответ.
Не придумывай новую информацию.
Перед ответом верни номер текста из которго используешь информацию, используй только число. Пример: 1.


## контекст
{search_context}

## вопрос
{prompt}

## ответ
"""

    response = client.chat(
        model="llama3.1:8b",
        messages=[
            {
                "role": "user",
                "content": prompt2,
            },
        ],
        options={
            "temperature": 0.0,
            "seed": 808042,
            "top_k": 20,
            "top_p": 0.9,
            "min_p": 0.0,
            "tfs_z": 0.5,
        },
    )
    answer = response["message"]["content"]
    
    answer_lines = answer.split("\n")
    # print(answer_lines)
    # print(prompt2)
    # print(answer)
    pprint(response["message"])
    res_id = len(os.listdir(path_to_save_prompts))
    with open(os.path.join(path_to_save_prompts,f"{res_id}.txt"),"w") as f:
        res = f"{prompt}\n< --- >\n{search_context}"
        f.write(res)
    number_of_texts = extract_integers_from_string(answer_lines[0])
    if -1 in number_of_texts:
        return "Нет информации."
    answer_text = "\n".join(answer_lines[1:])
    for n_text in number_of_texts:
        if n_text<5:
            answer_text += f'\nfilename: {search_res[0][n_text]["entity"]["file_name"]}.pdf page: {search_res[0][n_text]["entity"]["page"]}'
        else:
           answer_text = "E"

#     prompt2 = f"""
# ## Задание
# Ты специалист по работе с текстом. Кратко сформируй основную мысль полученого текста. Не упоминай номер текста.
# Не придумывай новые факты.
 
# ## текст
# {answer_text}

# ## ответ
# """
#     response = client.chat(
#         model="llama3.1:8b",
#         messages=[
#             {
#                 "role": "user",
#                 "content": prompt2,
#             },
#         ],
#         options={
#             "temperature": 0.0,
#             "seed": 808042,
#             "top_k": 20,
#             "top_p": 0.9,
#             "min_p": 0.0,
#             "tfs_z": 0.5,
#         },
#     )
#     print(prompt2)
#     # print(response["message"]["content"])
    # return response["message"]["content"]+file_names
    return answer_text


def chat_response(message, history):
    return test_inference(message, message)


def handle_file_upload(files:list[str]):
    for f in files:
        shutil.copy(f,os.path.join(patg_to_upload,f.split("/")[-1]))
    return f"Вы загрузили {len(files)} файл(ов)."


# with gr.Blocks() as iface1:
#     gr.ChatInterface(chat_response, type="messages")

with gr.Blocks() as iface2:
    file_input = gr.File(label="Выберите файлы", file_count="multiple")
    file_output = gr.Textbox(label="Результат загрузки")
    file_input.change(handle_file_upload, file_input, file_output)

demo = gr.TabbedInterface(
    [gr.ChatInterface(chat_response, type="messages"), iface2], ["bot", "file"]
)
if __name__ == "__main__":
    demo.launch(share=True)
