from typing import Union, Annotated

from .router import router

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, Depends
import firebase_admin
from pydantic import BaseModel

from .config import get_firebase_user_from_token, get_settings
from .session import ConversationManager

app = FastAPI()
app.include_router(router)
firebase_admin.initialize_app()

origins = get_settings().frontend_url_regex
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

convo = ConversationManager("weather_bot", 1)

import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.get("/session")
def update_item(user: Annotated[dict, Depends(get_firebase_user_from_token)]):
    convo_id = convo.new_conversation(user["uid"])
    return {"id": user["uid"], "convo_id": convo_id}

@app.put("/session/{conversation_id}")
async def update_item(user: Annotated[dict, Depends(get_firebase_user_from_token)], conversation_id: str, message: Message):
    response = await convo.push_conversation(conversation_id, user["uid"], message.text)
    print(message, " Responded with: ", response)
    return {"id": user["uid"], "response": response}

@app.put("/session/{conversation_id}/close")
def update_item(user: Annotated[dict, Depends(get_firebase_user_from_token)], conversation_id: str):
    convo.close_conversation(conversation_id, user["uid"])
    return {"id": user["uid"], "closed": True}