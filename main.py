from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel
from contextlib import asynccontextmanager

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

# Initialize browser agent
browser_agent = BrowserAgent()


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
