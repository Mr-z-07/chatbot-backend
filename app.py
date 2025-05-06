import os
import logging
from typing import List, Dict
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware
from minigpt import MiniGPT

# Load environment variables from .env file
load_dotenv()

# Fetch API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Ensure the API key is set in the environment
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Initialize FastAPI app with title and description
app = FastAPI(
    title="Chatbot Backend API",
    description="API for interacting with the chatbot",
    version="1.0.0"
)

# Add middleware for handling JSON responses
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route for API documentation
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Chatbot Backend API",
        "documentation": "Visit /docs for API documentation",
        "status": "running",
        "endpoints": {
            "chat": "/chat/"
        },
        "timestamp": datetime.now().isoformat()
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Set up CORS middleware to allow all origins (for development purposes)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize Groq client with API key
client = Groq(api_key=GROQ_API_KEY)

# Initialize MiniGPT
minigpt = MiniGPT()

# Set up logging for error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic model for the user input in chat
class User(BaseModel):
    message: str
    role: str = "user"  # Default role is 'user'
    conversation_id: str = None  # Optional conversation ID
    mode: str = "groq"  # Default mode is 'groq', can be 'minigpt'

# Conversation class to manage messages and active status
class Conversation:
    def __init__(self):
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        self.active = True

    def end_conversation(self):
        """Deactivate the conversation."""
        self.active = False

# Dictionary to store conversations using conversation_id
conversations: Dict[str, Conversation] = {}

# Function to query the Groq API and get a response
def query_groq_api(conversation: Conversation) -> str:
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",  # Updated model
            messages=conversation.messages,
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )
        
        # Concatenate the response chunks
        response = ""
        for chunk in completion:
            response += chunk.choices[0].delta.content or ""

        return response
    except Exception as e:
        logger.error(f"Error querying Groq API: {e}")
        raise HTTPException(status_code=500, detail=f"Error querying Groq API: {str(e)}")

# Function to get or create a conversation based on conversation_id
def get_or_create_conversation(conversation_id: str) -> Conversation:
    if conversation_id not in conversations:
        conversations[conversation_id] = Conversation()
    return conversations[conversation_id]

# Handle favicon requests
@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

# FastAPI endpoint for handling chat requests
@app.post("/chat/")
async def chat(input: User):
    try:
        # If using minigpt mode, use the local model instead of Groq API
        if input.mode == "minigpt":
            # Get response from MiniGPT
            response = minigpt.get_response(input.message)
            
            return {
                "response": response,
                "conversation_id": input.conversation_id
            }
        else:  # Default to groq mode
            # Get or create a conversation using the provided conversation_id
            conversation = get_or_create_conversation(input.conversation_id)

            # Check if the conversation is still active
            if not conversation.active:
                raise HTTPException(
                    status_code=400, detail="Conversation is inactive. Please start a new conversation."
                )

            # Append the user's message to the conversation
            conversation.messages.append({"role": input.role, "content": input.message})

            # Query Groq API for a response
            response = query_groq_api(conversation)

            # Append the assistant's response to the conversation
            conversation.messages.append({"role": "assistant", "content": response})

            return {
                "response": response,
                "conversation_id": input.conversation_id
            }

    except Exception as e:
        # Log the error and return an HTTP 500 response
        logger.error(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

# Main entry point to run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))  # Use port 10000 as default
    uvicorn.run(app, host="0.0.0.0", port=port)