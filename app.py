from fastapi import FastAPI
from pydantic import BaseModel
import ML_Model  # Import the generated script from Jupyter notebook

app = FastAPI()

# Define the expected input data schema based on the model inputs
class InputData(BaseModel):
    cost_plus_revenue: float
    per_unit_revenue: float
    fls_contribution: float
    bpc_contribution: float
    others_contribution: float
    storage_rev: float
    handling_revenue: float

@app.post("/predict")
async def predict(data: InputData):
    # Prepare the inputs for the model
    inputs = [[
        data.cost_plus_revenue,
        data.per_unit_revenue,
        data.fls_contribution,
        data.bpc_contribution,
        data.others_contribution,
        data.storage_rev,
        data.handling_revenue
    ]]
    
    # Use the model's prediction function
    try:
        prediction = ML_Model.predict(inputs)
    except Exception as e:
        return {"error": str(e)}
    
    # Return the prediction result
    return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
