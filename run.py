from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any
import json
from test import details
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/query")
async def Agent1(request: PromptRequest):
    prompt = request.prompt
    search_result = details(prompt)
    return search_result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# prompt = "I am X, a second year UG in iiit manipur .Give me the scholarships I'm eligible for."
# details(prompt)
