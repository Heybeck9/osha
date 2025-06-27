from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI(title="Converted API", description="Converted from Flask to FastAPI")

# Pydantic models
class User(BaseModel):
    name: str
    email: Optional[str] = ""

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

class Item(BaseModel):
    name: str
    price: Optional[float] = 0.0

class ItemResponse(BaseModel):
    id: int
    name: str
    price: float

# Simple in-memory storage
users = []
items = []

@app.get("/")
async def home():
    return {"message": "Welcome to the FastAPI API"}

@app.get("/users", response_model=List[UserResponse])
async def get_users():
    return users

@app.post("/users", response_model=UserResponse)
async def create_user(user: User):
    new_user = {
        "id": len(users) + 1,
        "name": user.name,
        "email": user.email
    }
    users.append(new_user)
    return new_user

@app.get("/items", response_model=List[ItemResponse])
async def get_items():
    return items

@app.post("/items", response_model=ItemResponse)
async def create_item(item: Item):
    new_item = {
        "id": len(items) + 1,
        "name": item.name,
        "price": item.price
    }
    items.append(new_item)
    return new_item

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    user = next((u for u in users if u['id'] == user_id), None)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
