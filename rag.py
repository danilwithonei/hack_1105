import re
from pymilvus import MilvusClient
from ollama import Client
from sentence_transformers import SentenceTransformer
import templates

class RAG:
    def __init__(
        self,
        ollama_host: str,
        milvus_uri: str = "./milvus_demo.db",
        sentence_transformer: str = "intfloat/multilingual-e5-large",
    ) -> None:
        self.ollama_client = Client(host=ollama_host)
        self.model = SentenceTransformer(sentence_transformer)

        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.collection_name = "my_rag_collection"

    def llm_inference(self, template, question, context=""):
        response = self.ollama_client.chat(
            model="llama3.1:8b",
            messages=[
                {
                    "role": "user",
                    "content": template.format(question=question, context=context),
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
        return answer

    def ask_db(self, template, question, context=""):
        search_result = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[
                self.emb_text(question)
            ],  # Use the `emb_text` function to convert the question to an embedding vector
            limit=5,  # Return top 5 results
            search_params={"metric_type": "IP", "params": {}},  # Inner product distance
            output_fields=["text", "file_name", "page"],  # Return the text field
        )[
            0
        ]  # mb batch
        search_context = ""
        for i, res in enumerate(search_result):
            search_context += f'текст № {i}:\n {res["entity"]["text"]}\n'
        numbers_of_texts = self.llm_inference(
            template=templates.text_number, question=question, context=search_context
        )
        numbers_of_texts = self.extract_integers_from_string(numbers_of_texts)
        if -1 in numbers_of_texts or len(numbers_of_texts) == 0:
            return "error -1"
        answer_text = self.llm_inference(
            template=template, question=question, context=search_context
        )

        for n_text in numbers_of_texts:
            if n_text < 5:
                answer_text += f'\nfilename: {search_result[n_text]["entity"]["file_name"]}.pdf page: {search_result[n_text]["entity"]["page"]}'
            else:
                answer_text = "E"
        return answer_text

    def classify(self):
        pass

    def extract_integers_from_string(self, s: str):
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

    def emb_text(self, text: str):
        embeddings = self.model.encode(text)
        return embeddings

    def milvus_search(self, text: str):
        search_result = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[
                self.emb_text(text)
            ],  # Use the `emb_text` function to convert the question to an embedding vector
            limit=5,  # Return top 5 results
            search_params={"metric_type": "IP", "params": {}},  # Inner product distance
            output_fields=["text", "file_name", "page"],  # Return the text field
        )[
            0
        ]  # mb batch

        search_context = ""
        for i, res in enumerate(search_result):
            search_context += f'текст № {i}:\n {res["entity"]["text"]}\n'
        return search_context

    # Перед ответом верни номер текста из которго используешь информацию, используй только число. Пример: 1.
    def get_prompt(self, milvus_output: str, question: str):
        prompt = f"""
## Задание
Ты специалист по работе с данными, ответь на вопрос используя контекст. Дай краткий ответ.
Не придумывай новую информацию.


## контекст
{milvus_output}

## вопрос
{question}

## ответ
"""
        return prompt

    # def get_

    def get_response(self, question: str, number_max_response: int = 5):
        milvus_output = self.milvus_search(question)
        prompt = self.get_prompt(milvus_output, question)
        response = self.ollama_client.chat(
            model="llama3.1:8b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
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
        number_of_texts = self.extract_integers_from_string(answer_lines[0])
        print("________question__________")
        print(question)
        print("________answer__________")
        print(answer)
        print("________milvus_output_____")
        print(milvus_output)

        # answer_text = "\n".join(answer_lines[1:])

        # for n_text in number_of_texts:
        #     if n_text < number_max_response:
        #         answer_text += f'\nfilename: {milvus_output[0][n_text]["entity"]["file_name"]}.pdf page: {milvus_output[0][n_text]["entity"]["page"]}'
        #     else:
        #         answer_text = "E"
        # print(answer_text)
        return answer
