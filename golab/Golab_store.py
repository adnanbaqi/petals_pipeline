from fastapi import APIRouter, HTTPException, status, File, UploadFile
from fastapi.responses import JSONResponse
from immudb.client import ImmudbClient
import re

router = APIRouter()

# Initialize immudb client
client = ImmudbClient()
client.login("Test", "Adnan123!@#")  # Use environment variables for credentials

# Mapping of file extensions to programming languages
LANGUAGE_MAP = {
    ".py": "Python",
    ".js": "JavaScript",
    ".java": "Java",
    ".cpp": "C++",
    ".cs": "C#",
    ".rb": "Ruby",
    ".go": "Go",
    ".php": "PHP",
    ".ts": "TypeScript",
    ".html": "HTML",
    ".css": "CSS",
    ".sql": "SQL",
    ".txt": "TextExtension-Based_Unknown ",
    # Add more mappings as needed
}

def count_tokens(text):
    token_pattern = r'\b\w+\b|\d+|[+\-*/=<>!&|%^~]+'
    tokens = re.findall(token_pattern, text)
    return len(tokens)

def detect_language(filename):
    # Extract the file extension and use it to infer the programming language
    extension = filename.split('.')[-1].lower()
    return LANGUAGE_MAP.get(f".{extension}", "Unknown")

@router.post("/api/v1/golab/store")
async def store(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")

        # Use the file name to detect the programming language
        language = detect_language(file.filename)

        num_tokens = count_tokens(text)

        key = f"{language}:{num_tokens}".encode()
        value = text.encode()
        client.set(key, value)

        return JSONResponse(content={"message": "Code snippet stored successfully", "language": language, "num_tokens": num_tokens}, status_code=status.HTTP_201_CREATED)
    except Exception as e:
        return JSONResponse(content={"detail": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
