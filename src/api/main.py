"""
AutoMind — FastAPI app entrypoint.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs (Swagger UI)
— genuinely useful to screen-record for your resume/portfolio demo.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from src.api.auth import authenticate_user, create_access_token
from src.api.routes import router as chat_router

app = FastAPI(
    title="AutoMind API",
    description="Agentic in-vehicle AI copilot — RAG over owner manuals + "
                "knowledge-graph fault reasoning, orchestrated with LangGraph.",
    version="0.1.0",
)

# Allow the Streamlit frontend (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this to your frontend's actual origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, tags=["chat"])


@app.post("/auth/login", tags=["auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_user(form_data.username, form_data.password):
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(form_data.username)
    return {"access_token": token, "token_type": "bearer"}


@app.get("/health", tags=["meta"])
def health_check():
    return {"status": "ok", "service": "AutoMind API"}