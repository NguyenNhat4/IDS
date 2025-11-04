from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes import detection, attack_simulator
import os

# Initialize FastAPI app
app = FastAPI(
    title="IDS Application",
    description="Intrusion Detection System with Machine Learning",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware (allow frontend to call backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

# Include routers
app.include_router(detection.router, prefix="/api", tags=["Detection"])
app.include_router(attack_simulator.router, prefix="/api", tags=["Simulator"])


@app.get("/")
async def root():
    """Serve frontend index.html"""
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "IDS Application API",
        "status": "running",
        "docs": "/api/docs",
        "frontend": "Frontend files not found. Please create frontend folder."
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "IDS Application"}


@app.get("/api")
async def api_info():
    """API information"""
    return {
        "name": "IDS Application API",
        "version": "1.0.0",
        "endpoints": {
            "detection": "/api/predict",
            "simulator": "/api/simulate/{attack_type}",
            "stats": "/api/stats",
            "docs": "/api/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting IDS Application...")
    print("📊 API Docs: http://localhost:8000/api/docs")
    print("🌐 Frontend: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
