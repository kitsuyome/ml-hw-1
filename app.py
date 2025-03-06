from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle
import io
from fastapi.responses import StreamingResponse

app = FastAPI()

with open("model_data.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item):
    data = pd.DataFrame([item.dict()])
    prediction = model.predict(data)[0]
    return {"predicted_price": prediction}

@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df.drop(columns=["selling_price"], inplace=True)
    predictions = model.predict(df.select_dtypes(include=['float64', 'int64']))
    df["predicted_price"] = predictions
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(stream, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=predictions.csv"})