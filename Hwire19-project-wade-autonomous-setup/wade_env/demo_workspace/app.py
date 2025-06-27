from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="WADE Demo API", description="Autonomous development demo")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None

items_db = []

@app.get("/")
async def root():
    return FileResponse('static/index.html')

@app.get("/api/status")
async def status():
    return {"message": "WADE Autonomous Development Demo", "items_count": len(items_db)}

@app.get("/items", response_model=List[Item])
async def get_items():
    return items_db

@app.post("/items", response_model=Item)
async def create_item(item: Item):
    items_db.append(item)
    return item

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id >= len(items_db):
        raise HTTPException(status_code=404, detail="Item not found")
    return items_db[item_id]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
