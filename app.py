from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from controller.store_data import router as store_router
from controller.load_data  import router as load_router  # Importing from the controller directory

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
)



app.include_router(store_router)
app.include_router(load_router)

@app.get("/", response_class=HTMLResponse)
async def get_home():
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Message</title>
    <style>
        body {
            background-color: black;
            color: white;
            height: 100vh;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
</head>
<body>
    <h1>Good To Go Chief...!</h1>
</body>
</html>
'''
