import re
from pymilvus import MilvusClient
from ollama import Client
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import datetime

# Наши прекрасные модули
import templates
from settings import *


class RAG:
    def __init__(
        self,
        ollama_host: str,
        milvus_uri: str = "./milvus_demo.db",
        sentence_transformer: str = "intfloat/multilingual-e5-large",
        db_docs_limit : int = 5,
        debug=False
    ) -> None:
        self.ollama_client = Client(host=ollama_host)
        self.model = SentenceTransformer(sentence_transformer)
        self.reranker_model = CrossEncoder('dangvantuan/CrossEncoder-camembert-large', max_length=128, device='cuda')

        self.milvus_client = MilvusClient(uri=milvus_uri)
        self.collection_name = "my_rag_collection"
        self.db_docs_limit = db_docs_limit
        self.debug = debug


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
        if self.debug:
            print(answer)
        return answer


    def ask_db(self, template, question, context=""):
        search_result = self.milvus_client.search(
            collection_name=self.collection_name,
            data=[
                self.emb_text(question)
            ],  # Use the `emb_text` function to convert the question to an embedding vector
            limit=self.db_docs_limit,  # Return top 5 results
            search_params={"metric_type": "IP", "params": {}},  # Inner product distance
            output_fields=["text", "file_name", "page"],  # Return the text field
        )[
            0
        ]  # mb batch
        search_context = ""
        documents = []
        for i, res in enumerate(search_result):
            documents.append(res["entity"]["text"])

        file_names = []
        best_file_info = []
        rank_result = self.reranker_model.rank(question, documents)
        
        last_score = -1
        for element in rank_result:
            ind = element['corpus_id']
            score = element['score']
            if score != -1 and last_score - score > 0.1:
                break
            if score > 0.009:
                name = templates.files_template.format(filename=search_result[ind]["entity"]["file_name"], page=search_result[ind]["entity"]["page"])
                is_ok = self.llm_inference(template=templates.has_answer_prompt, question=question, context=documents[ind])
                is_ok = self.extract_integers_from_string(is_ok)
                if len(is_ok) == 1:
                    if not(name in file_names) and is_ok[0] == 1:
                        search_context += f'Текст №{i}\n {res["entity"]["text"]}\n'
                        file_names.append(name)
                        if len(best_file_info) == 0:
                            best_file_info.append(search_result[ind]["entity"]["file_name"])
                            best_file_info.append(search_result[ind]["entity"]["page"])
                last_score = score

        if len(file_names) == 0:
            return templates.error_no_data, templates.error_no_data, None, None
        
        short_answer = self.llm_inference(template=template, question=question, context=search_context)

        answer_text = templates.db_answer_template.format(answer=short_answer, files=''.join(file_names))
        return short_answer, answer_text, best_file_info[0], best_file_info[1]


    def extract_integers_from_string(self, s: str):
        numbers = re.findall(r"-?\d+", s)
        integers = [int(num) for num in numbers]
        return integers


    def emb_text(self, text: str):
        embeddings = self.model.encode(text)
        return embeddings

    
    def extract_year(self, question):
        year = self.llm_inference(template=templates.year_extraction, question=question)
        year = self.extract_integers_from_string(year)
        if len(year) == 0:
            year = datetime.datetime.now().year
        else:
            year = year[0]
        return year

    
    def tree(self, question):
        if self.debug:
            print('Classification')
        scenario = self.llm_inference(template=templates.classifier, question=question)
        scenario = self.extract_integers_from_string(scenario)
        if len(scenario) == 0:
            return templates.error_question, '', ''
        
        scenario = [0]
        
        if scenario[0] == 0:
            short_answer, answer, filename, slide_number = self.ask_db(templates.base_prompt, question)
            return short_answer, answer, filename, slide_number

        elif scenario[0] == 1:
            company = self.llm_inference(template=templates.company_extraction, question=question)
            if 'null' in company:
                return templates.error_report, '', ''
            year = self.extract_year(question)
            
            report = templates.report_header.format(company=company, year=year)
            for field, field_question in templates.report.items():
                field_answer = self.ask_db(templates.base_prompt, field_question.format(company=company, year=year))
                report = '{}\n{}\n{}'.format(report, field, field_answer)
            return '', report, '', ''   

        elif scenario[0] == 2:
            answer = self.llm_inference(template=templates.about_company_prompt, question=question)
            return answer, answer, '', ''
        
        elif scenario[0] == 3:
            company = self.llm_inference(template=templates.company_extraction, question=question)
            if company == 'null':
                short_answer, answer, filename, slide_number = self.ask_db(templates.base_prompt, question)
                return short_answer, answer, filename, slide_number

            year = self.extract_year(question)
            if not(str(year) in question):
                question = '{} за {} год'.format(question, year)
            short_answer, answer, filename, slide_number = self.ask_db(templates.base_prompt, question)
            return short_answer, answer, filename, slide_number
        
        else:
            return templates.error_question, templates.error_question, '', ''
               


if __name__ == '__main__':
    rag = RAG(ollama_host = HOST, milvus_uri = MILVUS_DB, debug = True)

    text = ''

    while True:
        text = input('Enter your question:\n')

        _, answer, _, _ = rag.tree(text)

        print('Answer:\n{}'.format(answer))
        print('_' * 20)
