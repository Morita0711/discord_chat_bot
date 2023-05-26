import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain

app = FastAPI()
vectorstore: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore.pkl").exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    sender: str
    message: str
    type: str


@app.post("/chat")
async def chat(chat_request: ChatRequest):
    question = chat_request.question
    question_handler = QuestionGenCallbackHandler(None)
    stream_handler = StreamingLLMCallbackHandler(None)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)

    try:
        response = await qa_chain.acall({"question": question, "chat_history": chat_history})
        chat_history.append((question, response["answer"]))
        response_message = response["answer"]
        chat_response = ChatResponse(sender="bot", message=response_message, type="response")
        return JSONResponse(content=chat_response.dict())
    except Exception as e:
        logging.error(e)
        response_message = "Sorry, something went wrong. Try again."
        chat_response = ChatResponse(sender="bot", message=response_message, type="error")
        return JSONResponse(content=chat_response.dict())


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)

    while True:
        try:
            # Receive and send back the client message
            chat_request = await websocket.receive_json()
            question = chat_request.get("question")
            response_message = chat_request.get("response_message")
            chat_history.append((question, response_message))
            
            response = await qa_chain.acall({"question": question, "chat_history": chat_history})
            chat_history.append((question, response["answer"]))

            chat_response = ChatResponse(sender="bot", message=response["answer"], type="response")
            await websocket.send_json(chat_response.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            chat_response = ChatResponse(sender="bot", message="Sorry, something went wrong. Try again.", type="error")
            await websocket.send_json(chat_response.dict())


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

