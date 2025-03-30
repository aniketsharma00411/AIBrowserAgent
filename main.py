from fastapi import FastAPI, HTTPException
from playwright.async_api import async_playwright
from typing import Optional, Dict, Any, List
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from playwright_stealth import stealth_async
import json
import time
import base64
from io import BytesIO
import datetime
from contextlib import asynccontextmanager

# Load environment variables
load_dotenv()


class Command(BaseModel):
    command: str


class BrowserAgent:
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

    async def start(self):
        self.playwright = await async_playwright().start()
        # Launch browser with stealth mode
        self.browser = await self.playwright.chromium.launch(
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-features=IsolateOrigins,site-per-process',
                '--disable-site-isolation-trials',
                '--disable-web-security',
                '--disable-features=IsolateOrigins',
                '--disable-site-isolation-trials'
            ]
        )

        # Create context with stealth settings
        self.context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            java_script_enabled=True,
            bypass_csp=True,
            ignore_https_errors=True,
            extra_http_headers={
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Cache-Control': 'max-age=0'
            }
        )

        # Create page and apply stealth
        self.page = await self.context.new_page()
        await stealth_async(self.page)

    async def stop(self):
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def get_page_info(self) -> Dict[str, Any]:
        """Gather minimal information about the current page state with screenshot."""
        try:
            # Get the current URL and title
            current_url = self.page.url
            title = await self.page.title()

            # Take a screenshot
            if not os.path.exists("screenshots"):
                os.makedirs("screenshots")
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshots/screenshot_{timestamp}.png"
            page_screenshot = await self.page.screenshot(
                path=filename
            )

            # Convert screenshot to base64
            screenshot_base64 = base64.b64encode(
                page_screenshot).decode('utf-8')

            return {
                "url": current_url,
                "title": title,
                "screenshot": screenshot_base64
            }
        except Exception as e:
            return {"error": str(e)}

    async def execute_command(self, command: str) -> Dict[str, Any]:
        try:
            conversation_history = []
            page_info = None
            run_agent = True
            max_iterations = 10
            iteration = 0
            while iteration < max_iterations and run_agent:
                iteration += 1

                # Prepare messages for OpenAI
                messages = [
                    {
                        "role": "system",
                        "content": """
                        You are a browser automation assistant. Parse the user's command into a json format. You are allowed to perform an action and then request to see the page information before proceeding. Set the needs_page_info to true if you need to see the page information after performing the action before proceeding to take the next action. Your response should be parsable using Python's json.loads function.
                        The json should have the following fields:
                        - action: The action to perform. The action should be one of the following:
                            - no_action: Do nothing. Use it if you only need to see the page information screenshot.
                            - navigate: Navigate to the url, where the "url" is the key in the json. The value of the key will be the input to the goto function of playwright.
                            - click: Click on the element, where the "selector" is the key in the json. The value of the key will be the input to the click function of playwright.
                            - type: Type the text, where the "selector" and "text" are the keys in the json. The values of the keys will be the input to the fill function of playwright.
                            - search: Search for the query. For this we will have the following keys in the json. 
                                - url: The url to navigate to.  The value of the key will be the input to the goto function of playwright.
                                - selector: Click on the element. The value of the key will be the input to the fill function of playwright.
                                - text: The text to type. The value of the key will be the input to the fill function of playwright.
                        - needs_page_info: A boolean indicating whether you need to see the current page state before proceeding. If the user command has been executed, set this to false. Only set this to true if the user command has not been executed, confirm this from the screenshot.

                        Extra Information:
                        - If you are trying to input something at google.com then the selector is "textarea[name="q"]"
                        """
                    }
                ]

                # Add the user's command
                messages.append({"role": "user", "content": command})

                # Add conversation history with up to 5 messages
                for msg in conversation_history[-5:]:
                    messages.append(msg)

                print(messages)

                # Add page information if provided
                if page_info:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Current URL: {page_info['url']}\nCurrent Title: {page_info['title']}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{page_info['screenshot']}",
                                    "detail": "low"
                                }
                            }
                        ]
                    })

                # Get response from OpenAI
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )
                conversation_history.append({
                    "role": "assistant",
                    "content": response.choices[0].message.content.strip()
                })

                # Parse the response content
                try:
                    content = response.choices[0].message.content.strip()
                    print("Raw response:", content)  # Debug print
                    parsed_command = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {str(e)}")
                    print(f"Invalid JSON content: {content}")
                    conversation_history.append({
                        "role": "user",
                        "content": f"Failed to parse AI response: {str(e)}"
                    })
                    continue

                # Execute the parsed command
                try:
                    if parsed_command.get("action") == "navigate":
                        await self.page.goto(parsed_command["url"])
                    elif parsed_command.get("action") == "click":
                        await self.page.click(parsed_command["selector"])
                    elif parsed_command.get("action") == "type":
                        await self.page.fill(parsed_command["selector"], parsed_command["text"])
                    elif parsed_command.get("action") == "search":
                        await self.page.goto(parsed_command["url"])
                        await self.page.fill(parsed_command["selector"], parsed_command["text"])
                        await self.page.keyboard.press("Enter")
                        await self.page.wait_for_selector("div#search")
                except Exception as e:
                    print(f"Command execution error: {str(e)}")
                    conversation_history.append({
                        "role": "user",
                        "content": f"Failed to execute command: {str(e)}"
                    })
                    continue

                # Check if we need page information
                if parsed_command["needs_page_info"]:
                    page_info = await self.get_page_info()
                else:
                    run_agent = False

            return {
                "status": "success",
                "message": f"Executed command: {command}",
                "conversation_history": conversation_history
            }

        except Exception as e:
            print(f"General error: {str(e)}")
            return {"status": "error", "message": str(e)}


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

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
