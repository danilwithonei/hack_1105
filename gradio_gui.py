import gradio as gr
from rag import RAG
from settings import *
import shutil
import os
from milvus_tools import add_data

patg_to_upload = "upload_data"
path_to_save_prompts = "results"

rag = RAG(ollama_host = HOST, milvus_uri = MILVUS_DB, debug = False)


def handle_file_upload(files: list[str]):
    for f in files:
        shutil.copy(f, os.path.join(patg_to_upload, f.split("/")[-1]))
    add_data()
    return f"Вы загрузили {len(files)} файл(ов)."


def bot_response(message, history):
    _, answer, _, _ = rag.tree(message)
    return answer


with gr.Blocks() as iface1:
    gr.ChatInterface(bot_response,type="messages")

with gr.Blocks() as iface2:
    file_input = gr.File(label="Выберите файлы", file_count="multiple")
    file_output = gr.Textbox(label="Результат загрузки")
    file_input.change(handle_file_upload, file_input, file_output)

demo = gr.TabbedInterface(
    [iface1, iface2],
    ["bot", "file"],
)
demo.height = 1000
if __name__ == "__main__":
    demo.launch(share=True)
