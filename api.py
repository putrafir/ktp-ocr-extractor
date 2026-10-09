from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.pipeline import KTPExtractionPipeline

app = FastAPI(
    title="KTP Information Extractor API",
    description="API for local Indonesian KTP extraction, layout analysis, and document validation using PaddleOCR ONNX Runtime.",
    version="1.2.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline instance
pipeline = KTPExtractionPipeline()

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "KTP Information Extractor & Validator",
        "version": "1.2.0",
        "ocr_engine": "PaddleOCR (rapidocr-onnxruntime)",
        "validation_enabled": True
    }

@app.post("/extract")
async def extract_ktp(
    file: UploadFile = File(..., description="KTP image file (JPG, PNG, HEIC, WEBP)"),
    strict: bool = Query(True, description="Strict KTP validation (rejects non-KTP images with 422)")
):
    """
    Extracts key-value pair information from uploaded KTP image.
    Validates whether the uploaded image is an Indonesian KTP.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        result = pipeline.process(content, strict_validation=strict)

        if result.get("status") == "rejected":
            return JSONResponse(
                status_code=422,
                content={
                    "success": False,
                    "status": "rejected",
                    "filename": file.filename,
                    "message": "Dokumen yang diunggah bukan KTP Republik Indonesia yang valid.",
                    "validation": result.get("validation", {}),
                    "metadata": {
                        "engine": result["engine"],
                        "latency_ms": result["latency_ms"]
                    }
                }
            )

        return {
            "success": True,
            "status": "success",
            "filename": file.filename,
            "is_ktp": result.get("is_ktp", True),
            "validation": result.get("validation", {}),
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
