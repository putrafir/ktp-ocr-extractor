from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Literal
import time

from src.pipeline import KTPExtractionPipeline
from src.models import KTPData

app = FastAPI(
    title="KTP Extractor API",
    description="High-performance Local KTP Information Extractor (PaddleOCR / LiteParse)",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache pipelines
pipelines = {
    "paddle": KTPExtractionPipeline(engine="paddle"),
    "liteparse": KTPExtractionPipeline(engine="liteparse")
}

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "KTP Information Extractor",
        "engines_available": ["paddle", "liteparse"],
        "docs_url": "/docs"
    }

@app.post("/extract")
async def extract_ktp(
    file: UploadFile = File(..., description="KTP image file (JPG, PNG, HEIC, WEBP)"),
    engine: Literal["paddle", "liteparse"] = Query("paddle", description="OCR engine to use")
):
    """
    Extracts key-value pair information from uploaded KTP image.
    Supports JPG, PNG, HEIC (iPhone), and WEBP.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        pipeline = pipelines.get(engine, pipelines["paddle"])
        result = pipeline.process(content)

        return {
            "success": True,
            "filename": file.filename,
            "data": result["data"],
            "metadata": {
                "engine": result["engine"],
                "latency_ms": result["latency_ms"],
                "text_elements_count": result["text_items_count"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
