import asyncio
import argparse
import websockets
import json
from src.chat_interface import chat_manager, process_message
from dotenv import load_dotenv


async def send_ws_message(websocket, message_type, content):
    """Send a message through WebSocket"""
    try:
        await websocket.send(json.dumps({
            "type": message_type,
            "content": content
        }))
    except Exception as e:
        print(f"Error sending WebSocket message: {str(e)}")


async def run_repeat_process(chat_id: str, websocket):
    """Run the repeat process for a specific chat with real-time updates"""
    try:
        # Get chat history
        chat_history = await chat_manager.get_chat_history(chat_id)
        user_messages = [msg for msg in chat_history if msg["role"] == "user"]

        # Create new chat
        new_chat_id = await chat_manager.create_chat()
        print(f"Created new chat: {new_chat_id}")

        # Process each message
        for i, message in enumerate(user_messages):
            # Show user message
            await send_ws_message(websocket, "user_message", message["content"])

            # Process the message
            await chat_manager.save_message(new_chat_id, "user", message["content"])
            result = await process_message(new_chat_id, message)

            # Show assistant response
            if result.get("response"):
                await send_ws_message(websocket, "assistant_message", result["response"])

            # Wait a bit between messages for better visibility
            await asyncio.sleep(1)

        print(f"Successfully completed repeat process for chat {chat_id}")
    except Exception as e:
        print(f"Error running repeat process for chat {chat_id}: {str(e)}")
        await send_ws_message(websocket, "error", str(e))


async def main(chat_id: str):
    # Load environment variables for other configs
    load_dotenv()

    # Connect to WebSocket
    uri = f"ws://localhost:8000/api/ws/chat/{chat_id}"
    async with websockets.connect(uri) as websocket:
        # Run the repeat process with WebSocket connection
        await run_repeat_process(chat_id, websocket)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Repeat a chat process')
    parser.add_argument('--chat_id', help='The ID of the chat to repeat')
    args = parser.parse_args()

    asyncio.run(main(args.chat_id))
