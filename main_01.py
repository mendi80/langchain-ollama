"""
Connect langchain, ollama, gemma2
Send an application/json request to the API endpoint of Ollama to interact:
curl http://localhost:11434/api/generate -d '{"model": "gemma2", "prompt":"Why is the sky blue?"}'
curl http://localhost:11434/api/generate -H "Content-Type: application/json" -d "{\"model\": \"gemma2\", \"prompt\":\"Why is the sky blue?\"}"
"""
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=unused-import
# pylint: disable=wrong-import-position

# %%
from langchain_community.llms import Ollama  # pip install langchain-community
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(model="gemma2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

# %%
# llm.invoke("Tell me 5 facts about Roman history:")
# for chunks in llm.stream("Tell me a joke"): print(chunks)


# %%
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

templates = Jinja2Templates(directory="templates")
app = FastAPI()


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    response: str


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/invoke")
def invoke(prompt: str):
    return ChatResponse(response=llm.invoke(prompt))

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        print(f"request.prompt={request.prompt}")
        return StreamingResponse(llm.stream(request.prompt), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
