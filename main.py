# Importing necessary libraries

import uvicorn

import pickle

from pydantic import BaseModel

from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

# Initializing the fast API server

app = FastAPI()

origins = [


"http://localhost",

"http://localhost:8080",

"http://localhost:3000",

]

app.add_middleware(

CORSMiddleware,

allow_origins=origins,

allow_credentials=True,

allow_methods=["*"],

allow_headers=["*"],

)
model = pickle.load(open('../model/hackOverflow.pkl', 'rb'))

class Candidate(BaseModel):

    LQR: int
    Hackathons: int
    CSR: int
    PSP: int
    SLC: str
    ECD: str
    Cer: str
    WS: str
    RAWS: str
    MCS: str
    IS: str
    ICA: str
    ToC: str
    IToB: str
    TI:str
    MT: str
    HWSW: str
    WiT: str
    I: str

@app.get("/")
def read_root(self):
    return {"data": "Welcome to online career prediction model"}


 # Setting up the prediction route
@app.post("/prediction/")
async def get_predict(data: Candidate):
    sample = [[
        data.LQR,
        data.Hackathons,
        data.CSR,
        data.PSP,
        data.SLC,
        data.ECD,
        data.Cer,
        data.WS,
        data.RAWS,
        data.MCS,
        data.IS,
        data.ICA,
        data.ToC,
        data.IToB,
        data.TI,
        data.MT,
        data.HWSW,
        data.WiT,
        data.I
    ]]
    position = model.predict(sample).tolist()[0]

    return {

        "data": {

            'prediction': position,
        }

    }
# Configuring the server host and port

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')









