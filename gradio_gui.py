import gradio as gr
from rag import RAG
from settings import host
import shutil
import os

patg_to_upload = "upload_data"
path_to_save_prompts = "results"

rag_system = RAG(ollama_host=host)


def handle_file_upload(files: list[str]):
    for f in files:
        shutil.copy(f, os.path.join(patg_to_upload, f.split("/")[-1]))
    return f"Вы загрузили {len(files)} файл(ов)."


with gr.Blocks() as iface2:
    file_input = gr.File(label="Выберите файлы", file_count="multiple")
    file_output = gr.Textbox(label="Результат загрузки")
    file_input.change(handle_file_upload, file_input, file_output)

demo = gr.TabbedInterface(
    [gr.ChatInterface(rag_system.get_response, type="messages"), iface2],
    ["bot", "file"],
)
if __name__ == "__main__":
    demo.launch(share=True)
