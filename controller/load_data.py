from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from immudb.client import ImmudbClient

router = APIRouter()

# Initialize immudb client
client = ImmudbClient()
client.login("Golden_Retriever", "Get123!@#") 

@router.get("/api/v1/load")
async def fetch_all():
    try:
        key = b''  # Starting key, adjust as needed
        prefix = b''  # Common prefix for keys, if applicable
        desc = False  # Ascending order
        limit = 1000  # Maximum number of entries to fetch, adjust based on your performance considerations
        
        # Perform the scan operation with the specified parameters
        all_data = client.scan(key, prefix, desc, limit)
        
        # Format and return the data
        formatted_data = [{"key": key.decode("utf-8"), "value": value.decode("utf-8")} for key, value in all_data.items()]
        return JSONResponse(content={"data": formatted_data}, status_code=status.HTTP_200_OK)
    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.get("/api/v1/load/{key}")
async def load(key: str):
    try:
        # Assuming `key` is already url-safe and correctly formatted
        response = client.get(key.encode())
        
        if response is None:
            raise HTTPException(status_code=404, detail="Key not found")
        
        # Access the correct attribute of the response object to get the byte string
        # This example assumes the attribute is named 'value', adjust as necessary
        byte_value = response.value  # Adjust this line based on the actual attribute name
        
        text = byte_value.decode("utf-8")
        return {"text": text}
    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

