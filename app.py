# Imports
from fastapi import FastAPI, Form, Request, Response, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder

from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

import os
import json
import uvicorn
import aiofiles
import csv
from PyPDF2 import PdfReader

# ✅ Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "YOUR API KEY"

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ✅ PDF Processing
def file_processing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()
    full_text = ''.join([page.page_content for page in data])

    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = splitter_ques_gen.split_text(full_text)
    document_ques_gen = [Document(page_content=chunk) for chunk in chunks]

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen


# ✅ LLM Pipeline using Gemini
def llm_pipeline(file_path):
    document_ques_gen, document_answer_gen = file_processing(file_path)

    llm_ques_gen_pipeline = ChatGoogleGenerativeAI(
        temperature=0.3,
        model="models/gemini-1.5-flash"
  # Only works if you use OpenAI keys instead

    )

    prompt_template = """
    You are an expert at creating questions based on study materials and reference guides.
    Your goal is to prepare a student or teacher for their exams tests.
    You do this asking questions about the text below:
    ------------
    {text}
    ------------
    Create questions that will prepare the student or teacher for their tests.
    Make sure not to lose any important information.
    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = """
    You are an expert at creating practice questions based on study materials and reference guides.
    Your goal is to prepare a student or teacher for their exams tests.
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------
    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS
    )

    ques = ques_gen_chain.run(document_ques_gen)

    # ✅ Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = ChatGoogleGenerativeAI(
        temperature=0.1,
        model="models/gemini-1.5-flash"
 # Only works if you use OpenAI keys instead

    )

    ques_list = [q.strip() for q in ques.split("\n") if q.strip().endswith("?") or q.strip().endswith(".")]

    answer_generation_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )

    return answer_generation_chain, ques_list


# ✅ CSV Writer
def get_csv(file_path):
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = 'static/output/'
    os.makedirs(base_folder, exist_ok=True)

    output_file = os.path.join(base_folder, "QA.csv")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Question", "Answer"])

        for question in ques_list:
            print("Question:", question)
            answer = answer_generation_chain.run(question)
            print("Answer:", answer)
            print("--------------------------------------------------\n")
            writer.writerow([question, answer])

    return output_file


# ✅ Routes
@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def upload_pdf(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    base_folder = 'static/docs/'
    os.makedirs(base_folder, exist_ok=True)
    pdf_filename = os.path.join(base_folder, filename)

    async with aiofiles.open(pdf_filename, 'wb') as f:
        await f.write(pdf_file)

    response_data = jsonable_encoder(json.dumps({"msg": "success", "pdf_filename": pdf_filename}))
    return Response(response_data)


@app.post("/analyze")
async def analyze_pdf(request: Request, pdf_filename: str = Form(...)):
    output_file = get_csv(pdf_filename)
    response_data = jsonable_encoder(json.dumps({"output_file": output_file}))
    return Response(response_data)


# ✅ Run server
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
