
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

app = FastAPI(title="PromptForge AI Service")

# Allow communication with React and Express
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"


# AI generation request
class PromptRequest(BaseModel):
    prompt: str
    model: str = "qwen2.5:3b"
    temperature: float = 0.7
    max_tokens: int = 512


# Evaluation request
class StrategyResponse(BaseModel):
    strategy: str
    response: str


class EvaluationRequest(BaseModel):
    responses: list[StrategyResponse]


# Home route
@app.get("/")
def home():
    return {
        "status": "success",
        "message": "PromptForge AI Python Service is running!"
    }


# Ollama health check
@app.get("/api/ollama-health")
def ollama_health():
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        response.raise_for_status()

        return {
            "status": "success",
            "message": "Ollama is connected!",
            "models": [
                model["name"]
                for model in response.json().get("models", [])
            ]
        }

    except requests.RequestException as error:
        return {
            "status": "error",
            "message": str(error)
        }


# Generate an AI response
@app.post("/api/generate")
def generate_response(request: PromptRequest):

    if not request.prompt.strip():
        return {
            "status": "error",
            "message": "Please enter a prompt."
        }

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": request.model,
                "prompt": request.prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            },
            timeout=300
        )

        response.raise_for_status()
        result = response.json()

        return {
            "status": "success",
            "model": request.model,
            "prompt": request.prompt,
            "generated_response": result.get("response", ""),
            "total_duration": result.get("total_duration")
        }

    except requests.RequestException as error:
        return {
            "status": "error",
            "message": f"Ollama generation failed: {error}"
        }


# Evaluate prompt strategy responses
@app.post("/api/evaluate")
def evaluate_responses(request: EvaluationRequest):

    valid_responses = [
        item for item in request.responses
        if item.response.strip()
    ]

    if len(valid_responses) < 2:
        return {
            "status": "error",
            "message": "At least two non-empty responses are required."
        }

    original = next(
        (
            item for item in valid_responses
            if item.strategy.lower() == "original"
        ),
        None
    )

    if original is None:
        return {
            "status": "error",
            "message": "An Original strategy response is required."
        }

    texts = [item.response for item in valid_responses]

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)

        original_index = valid_responses.index(original)

        results = []
        similarities = []

        for index, item in enumerate(valid_responses):

            similarity = float(
                cosine_similarity(
                    tfidf_matrix[original_index],
                    tfidf_matrix[index]
                )[0][0]
            )

            word_count = len(item.response.split())

            if index != original_index:
                similarities.append(similarity)

            results.append({
                "strategy": item.strategy,
                "word_count": word_count,
                "similarity_to_original": round(
                    similarity * 100, 2
                )
            })

        average_similarity = (
            sum(similarities) / len(similarities)
            if similarities else 1.0
        )

        sensitivity_score = (
            1 - average_similarity
        ) * 100

        return {
            "status": "success",
            "evaluation_method": "TF-IDF and Cosine Similarity",
            "total_responses": len(results),
            "average_similarity": round(
                average_similarity * 100, 2
            ),
            "prompt_sensitivity_score": round(
                sensitivity_score, 2
            ),
            "results": results
        }

    except ValueError as error:
        return {
            "status": "error",
            "message": f"Could not evaluate responses: {error}"
        }
