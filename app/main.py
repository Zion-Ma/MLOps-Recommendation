# creates FastAPI app
from fastapi import FastAPI

app = FastAPI(title="News Recommendation API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
