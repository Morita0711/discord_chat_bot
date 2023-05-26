
## ✅ Running locally
1. export OPENAI_API_KEY=sk-x...
1. Install dependencies: `pip install -r requirements.txt`
1. Run `ingest.sh` to ingest LangChain docs data into the vectorstore (only needs to be done once).
   1. You can use other [Document Loaders](https://langchain.readthedocs.io/en/latest/modules/document_loaders.html) to load your own data into the vectorstore.
1. Run the app: `make start`
   1. To enable tracing, make sure `langchain-server` is running locally and pass `tracing=True` to `get_chain` in `main.py`. You can find more documentation [here](https://langchain.readthedocs.io/en/latest/tracing.html).
1. Open [localhost:9000](http://localhost:9000) in your browser.

## ✅ Running As a Server. 
1. export OPENAI_API_KEY=sk-x...
1. Install dependencies: `pip install -r requirements.txt`
1. Run `ingest.sh` to ingest LangChain docs data into the vectorstore (only needs to be done once).
1. Run `uvicorn srv:app --reload` to start the API server. 
1. Send HTTP Post requests to the `/chat` route I.E:
```
curl -X POST -H "Content-Type: application/json" -d '{"question": "hello?"}' http://0.0.0.0:8080/chat
```
