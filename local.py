"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore
model_name = "deepset/roberta-base-squad2"

from callback import QuestionGenCallbackHandler
from schemas import ChatResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None
llm_model = model_name
llm_pipeline = pipeline('question-answering', model=model_name, tokenizer=model_name)

@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore and local LLM")
    if not Path("vectorstore.pkl").exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)

    global llm_model, llm_pipeline
    llm_model = AutoModelForCausalLM.from_pretrained("model_path")
    tokenizer = AutoTokenizer.from_pretrained("model_path")
    llm_pipeline = Pipeline(model=llm_model, tokenizer=tokenizer, task='text-generation')

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    chat_history = []
    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = llm_pipeline(question)
            chat_history.append((question, result))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
