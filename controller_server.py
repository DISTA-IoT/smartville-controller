import logging
from fastapi import FastAPI
import uvicorn
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("controller_server")

app = FastAPI(title="Controller Server API", description="API for all")



@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    logger.info(f"Item requested with id: {item_id}")
    return {"item_id": item_id, "timestamp": datetime.now().isoformat()}

@app.post("/items/")
async def create_item(item: dict):
    logger.info(f"Creating new item: {item}")
    return {"item": item, "created": True}

@app.get("/health")
async def health_check():
    logger.debug("Health check endpoint called")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    logger.info("Starting FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000)