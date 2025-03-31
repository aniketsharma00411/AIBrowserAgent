from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
import os
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src import chat_interface
from src.browser_agent import BrowserAgent

# Load environment variables
load_dotenv()


class Command(BaseModel):
    command: str


class ExtractionRequest(BaseModel):
    query: str  # Natural language query for extraction


# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize browser
    await browser_agent.start()
    yield
    # Shutdown: clean up browser
    await browser_agent.stop()


# Initialize FastAPI app with lifespan
app = FastAPI(title="Browser Automation AI Agent", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize browser agent
browser_agent = BrowserAgent()

# Inject browser agent into chat interface
chat_interface.set_browser_agent(browser_agent)

# Include the chat interface router
app.include_router(chat_interface.router, prefix="")

# Get the absolute path to the static directory
static_dir = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "src", "static")

# Mount static files
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.post("/api/interact")
async def interact(command: Command):
    try:
        result = await browser_agent.execute_command(
            command.command
        )

        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract")
async def extract(request: ExtractionRequest):
    """Extract data from the current page based on natural language query"""
    try:
        global browser_agent
        result = await browser_agent.extract(request.query)
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/new")
async def create_new_chat():
    """Create a new chat and redirect to it"""
    try:
        print("Creating new chat")
        chat_id = await chat_interface.chat_manager.create_chat()
        return RedirectResponse(url=f"/chat/{chat_id}")
    except Exception as e:
        return RedirectResponse(url="/")


@app.get("/chat/{chat_id}")
async def get_chat_page(chat_id: str):
    """Serve the chat interface for a specific chat ID"""
    try:
        # First validate if chat exists
        exists = await chat_interface.chat_manager.validate_chat_id(chat_id)
        if not exists:
            # If chat doesn't exist, redirect to new chat
            return RedirectResponse(url="/chat/new")

        # If chat exists, serve the page
        return FileResponse(os.path.join(static_dir, "index.html"))
    except Exception as e:
        # If there's any error, redirect to new chat
        return RedirectResponse(url="/chat/new")


@app.get("/")
async def get_root():
    """Redirect root to a new chat"""
    return RedirectResponse(url="/chat/new")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
