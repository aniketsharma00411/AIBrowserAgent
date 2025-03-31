from fastapi import APIRouter, WebSocket, HTTPException, WebSocketDisconnect
from pymongo import MongoClient
from typing import List, Dict, Any
import json
import os
from datetime import datetime
import logging
from src.browser_agent import BrowserAgent


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router instead of FastAPI app
router = APIRouter()

# MongoDB connection with error handling
mongodb_available = False
try:
    MONGODB_URL = os.getenv("MONGODB_URL")
    logger.info(f"Attempting to connect to MongoDB at {MONGODB_URL}")
    client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
    # Verify connection
    client.server_info()
    db = client.browser_automation
    chat_collection = db.chats
    mongodb_available = True
    logger.info("Successfully connected to MongoDB")
except Exception as e:
    logger.error(f"Error connecting to MongoDB: {str(e)}")
    logger.error("Please ensure MongoDB is running and accessible")
    client = None
    db = None
    chat_collection = None

# Initialize browser agent
browser_agent = BrowserAgent()


def set_browser_agent(agent: BrowserAgent):
    """Set the browser agent instance"""
    global browser_agent
    browser_agent = agent


class ChatManager:
    def __init__(self):
        self.active_chats = {}

    async def get_chat_history(self, chat_id: str) -> List[Dict[str, Any]]:
        if not mongodb_available:
            logger.error("MongoDB connection not available")
            raise HTTPException(
                status_code=503,
                detail="MongoDB is not available. Please ensure MongoDB is running."
            )
        try:
            chat = chat_collection.find_one({"chat_id": chat_id})
            if not chat:
                logger.error(f"Chat not found: {chat_id}")
                raise HTTPException(status_code=404, detail="Chat not found")
            logger.info(f"Retrieved chat history for ID: {chat_id}")
            return chat.get("messages", [])
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Database service unavailable"
            )

    async def save_message(self, chat_id: str, role: str, content: str):
        if not mongodb_available:
            logger.error("MongoDB connection not available")
            raise HTTPException(
                status_code=503,
                detail="MongoDB is not available. Please ensure MongoDB is running."
            )
        try:
            message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow().isoformat()
            }
            result = chat_collection.update_one(
                {"chat_id": chat_id},
                {"$push": {"messages": message}},
                upsert=True
            )
            if result.matched_count == 0 and not result.upserted_id:
                logger.error(f"Failed to save message for chat {chat_id}")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to save message"
                )
            logger.info(f"Saved message for chat ID: {chat_id}")
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error saving message: {str(e)}"
            )

    async def create_chat(self) -> str:
        if not mongodb_available:
            logger.error("MongoDB connection not available")
            raise HTTPException(
                status_code=503,
                detail="MongoDB is not available. Please ensure MongoDB is running."
            )
        try:
            chat_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            print(f"Creating new chat with ID: {chat_id}")
            chat = {
                "chat_id": chat_id,
                "messages": [],
                "created_at": datetime.utcnow().isoformat()
            }
            result = chat_collection.insert_one(chat)
            if not result.inserted_id:
                logger.error("Failed to create new chat")
                raise HTTPException(
                    status_code=500,
                    detail="Failed to create new chat"
                )
            logger.info(f"Created new chat with ID: {chat_id}")
            return chat_id
        except Exception as e:
            logger.error(f"Error creating chat: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error creating chat: {str(e)}"
            )

    async def validate_chat_id(self, chat_id: str) -> bool:
        if not mongodb_available:
            logger.error("MongoDB connection not available")
            raise HTTPException(
                status_code=503,
                detail="MongoDB is not available. Please ensure MongoDB is running."
            )
        try:
            chat = chat_collection.find_one({"chat_id": chat_id})
            exists = chat is not None
            logger.info(
                f"Validated chat ID {chat_id}: {'exists' if exists else 'not found'}")
            return exists
        except Exception as e:
            logger.error(f"Error validating chat ID: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error validating chat ID: {str(e)}"
            )


chat_manager = ChatManager()


@router.post("/api/chat/create")
async def create_chat():
    try:
        logger.info("Received request to create new chat")
        chat_id = await chat_manager.create_chat()
        return {"chat_id": chat_id}
    except HTTPException as he:
        logger.error(f"HTTP error creating chat: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error creating chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error creating chat: {str(e)}"
        )


@router.get("/api/chat/{chat_id}/history")
async def get_chat_history(chat_id: str):
    try:
        if not await chat_manager.validate_chat_id(chat_id):
            raise HTTPException(status_code=404, detail="Chat not found")
        messages = await chat_manager.get_chat_history(chat_id)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/chat/{chat_id}/message")
async def process_message(chat_id: str, message: Dict[str, str]):
    try:
        if not await chat_manager.validate_chat_id(chat_id):
            raise HTTPException(status_code=404, detail="Chat not found")

        if not browser_agent:
            raise HTTPException(
                status_code=503,
                detail="Browser agent not initialized. Please ensure the browser agent is set."
            )

        user_message = message.get("content", "")
        if not user_message:
            raise HTTPException(
                status_code=400, detail="Message content is required")

        # Save user message
        await chat_manager.save_message(chat_id, "user", user_message)

        # Get chat history for context
        browser_agent.chat_history = await chat_manager.get_chat_history(chat_id)

        # Prepare messages for GPT
        messages = [
            {
                "role": "system",
                "content": """
                You are a browser automation assistant. Analyze the user's message and determine whether to:
                1. Execute a browser action (using the execute_command API)
                2. Extract data from the current page (using the extract API)
                
                Respond with a JSON object containing:
                {
                    "action_type": "execute" or "extract",
                    "command": "the command to execute" or "the data to extract"; the command will be passed to another AI model to generate a json,
                    "explanation": "brief explanation of your decision"
                }
                """
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        # Get response from GPT using browser_agent's client
        response = browser_agent.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        try:
            decision = json.loads(response.choices[0].message.content.strip())

            # Execute the appropriate action
            if decision["action_type"] == "execute":
                result = await browser_agent.execute_command(decision["command"])
            else:  # extract
                result = await browser_agent.extract(decision["command"])

            # Save assistant's response
            await chat_manager.save_message(
                chat_id,
                "assistant",
                json.dumps({
                    "decision": decision,
                    "result": result
                })
            )

            return {
                "status": "success",
                "decision": decision,
                "result": result
            }

        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500, detail="Invalid response from GPT")
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error executing command: {str(e)}")

    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/api/ws/chat/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    try:
        if not await chat_manager.validate_chat_id(chat_id):
            await websocket.close(code=4004, reason="Chat not found")
            return

        await websocket.accept()
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Process message and get response
                response = await process_message(chat_id, message)

                # Send response back to client
                await websocket.send_text(json.dumps(response))

            except WebSocketDisconnect:
                print(f"WebSocket disconnected for chat {chat_id}")
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "status": "error",
                    "message": "Invalid message format"
                }))
            except Exception as e:
                print(f"Error processing message: {str(e)}")
                await websocket.send_text(json.dumps({
                    "status": "error",
                    "message": str(e)
                }))

    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.post("/api/chat/{chat_id}/repeat")
async def repeat_chat_process(chat_id: str):
    """Repeat the chat process by creating a new chat and processing user messages from the previous chat"""
    try:
        # Get chat history from the previous chat
        chat_history = await chat_manager.get_chat_history(chat_id)

        # Filter user messages
        user_messages = [msg for msg in chat_history if msg["role"] == "user"]

        if not user_messages:
            raise HTTPException(
                status_code=400, detail="No user messages to repeat")

        new_chat_id = await chat_manager.create_chat()

        # Return the new chat ID immediately
        return {
            "status": "success",
            "message": "New chat created. Messages will be processed interactively.",
            "new_chat_id": new_chat_id,
            "messages_to_process": len(user_messages)
        }

    except Exception as e:
        logger.error(f"Error repeating chat process: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/chat/{chat_id}/process_next_message")
async def process_next_message(chat_id: str, request: dict):
    """Process the next message from the source chat in the current chat"""
    try:
        # Get parameters from request body
        source_chat_id = request.get("source_chat_id")
        message_index = request.get("message_index")

        if not source_chat_id or message_index is None:
            raise HTTPException(
                status_code=400, detail="source_chat_id and message_index are required")

        # Get chat history from the source chat
        chat_history = await chat_manager.get_chat_history(source_chat_id)
        user_messages = [msg for msg in chat_history if msg["role"] == "user"]

        if message_index >= len(user_messages):
            return {
                "status": "complete",
                "message": "All messages have been processed"
            }

        # Get the next message to process
        message = user_messages[message_index]

        try:
            # Then process the message
            response = await process_message(chat_id, {"content": message["content"]})

            return {
                "status": "success",
                "message": f"Processed message {message_index + 1} of {len(user_messages)}",
                "response": response,
                "next_index": message_index + 1
            }
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                # Continue with next message even if this one failed
                "next_index": message_index + 1
            }

    except Exception as e:
        logger.error(f"Error in process_next_message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
