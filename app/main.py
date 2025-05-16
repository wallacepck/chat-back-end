from typing import Union, Annotated

from .router import router

from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, Depends
import firebase_admin
from pydantic import BaseModel

from .config import get_firebase_user_from_token, get_settings
from .session import ConversationManager, ConversationOverloadError, InvalidConversationError

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

convo = ConversationManager("weather_bot", 3)

import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.get("/session")
def new_session(user: Annotated[dict, Depends(get_firebase_user_from_token)]):
    try:
        convo.init_conversation(user["uid"]) 
        return {"id": user["uid"]}
    except ConversationOverloadError as e:
        return JSONResponse(content={'message': "Too many ongoing conversations, please try again later."}, 
                            status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.put("/session/talk")
async def push_convo(user: Annotated[dict, Depends(get_firebase_user_from_token)], message: Message):
    try:
        generator = await convo.generate_conversation(user["uid"], message.text)
        return StreamingResponse(
            generator, 
            media_type='text/event-stream'
        )
    except InvalidConversationError as e:
        return JSONResponse(content={'message': "Could not find an ongoing conversation for this user! Please init the conversation first."}, 
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.put("/session/close")
def close_convo(user: Annotated[dict, Depends(get_firebase_user_from_token)]):
    convo.close_conversation(user["uid"])
    return {"id": user["uid"], "closed": "true"}