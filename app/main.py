from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.v1.endpoints import analyze

app = FastAPI()


# Mount API routes
app.include_router(analyze.router, prefix="/v1/analyze")

# Mount frontend static files directory to serve index.html and other assets
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
