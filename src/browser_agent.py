from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
from typing import Optional, Dict, Any
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import json
import base64
import datetime
from pydantic import field_validator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='logs/browser_agent.log',
                    encoding='utf-8')
logger = logging.getLogger(__name__)


class ExtractParams(BaseModel):
    selector: str = Field(
        description="The CSS selector to use for extracting data")
    attribute: Optional[str] = Field(
        None, description="The attribute to extract (optional, if not provided, text content will be extracted)")
    multiple: bool = Field(
        False, description="Boolean indicating whether to get all matching elements or just the first one")
    explanation: str = Field(
        description="A brief explanation of why these parameters were chosen")


class BrowserCommand(BaseModel):
    action: str = Field(
        description="The action to perform (navigate, click, type, search, login, no_action)")
    needs_page_info: bool = Field(
        description="Whether page information is needed before proceeding")
    extract_data: Optional[str] = Field(
        None, description="Natural language description of data to extract")

    # Optional fields based on action type
    url: Optional[str] = Field(None, description="URL to navigate to")
    selector: Optional[str] = Field(
        None, description="CSS selector for the element")
    text: Optional[str] = Field(
        None, description="Text to type into the field")
    username_selector: Optional[str] = Field(
        None, description="Selector for username field")
    password_selector: Optional[str] = Field(
        None, description="Selector for password field")
    username: Optional[str] = Field(None, description="Username to enter")
    password: Optional[str] = Field(None, description="Password to enter")
    submit_selector: Optional[str] = Field(
        None, description="Selector for submit button")

    @field_validator('action')
    def validate_action(cls, v):
        valid_actions = ['navigate', 'click',
                         'type', 'search', 'login', 'no_action']
        if v not in valid_actions:
            raise ValueError(f"Action must be one of {valid_actions}")
        return v


class BrowserAgent:
    def __init__(self):
        # Initialize LangChain model
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1
        )
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.chat_history = []

        # Setup JsonOutputParser for structured outputs
        self.command_parser = JsonOutputParser(pydantic_object=BrowserCommand)
        self.extract_parser = JsonOutputParser(pydantic_object=ExtractParams)

        logger.info("BrowserAgent initialized with LangChain integration")

    async def start(self):
        try:
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
            logger.info("Browser session started successfully")
        except Exception as e:
            logger.error(f"Failed to start browser session: {str(e)}")
            raise

    async def stop(self):
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def get_page_info(self, extract_data: str | None = None) -> Dict[str, Any]:
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

            data = None
            if extract_data:
                extract_result = await self.extract(extract_data)
                if extract_result["status"] == "success":
                    data = extract_result["data"][:50]

            return {
                "url": current_url,
                "title": title,
                "screenshot": screenshot_base64,
                "data": data
            }
        except Exception as e:
            return {"error": str(e)}

    async def execute_command(self, command: str) -> Dict[str, Any]:
        try:
            page_info = None
            run_agent = True
            max_iterations = 10
            iteration = 0

            while iteration < max_iterations and run_agent:
                iteration += 1
                logger.info(
                    f"Command iteration {iteration}: Processing '{command}'")

                # Prepare messages for LangChain
                messages = [
                    SystemMessage(content="""
                    You are a browser automation assistant. Parse the user's command into a structured format.
                    You are allowed to perform an action and then request to see the page information before proceeding.
                    Set the needs_page_info to true if you need to see the page information after performing the action
                    before proceeding to take the next action.

                    The structured output should have the following fields:
                    - action: The action to perform. The action should be one of the following:
                        - no_action: Do nothing. Use it if you only need to see the page information screenshot.
                        - navigate: Navigate to the url, where the "url" is the key in the json. The value of the key will be the input to the goto function of playwright.
                        - click: Click on the element, where the "selector" is the key in the json. The value of the key will be the input to the click function of playwright.
                        - type: Type the text, where the "selector" and "text" are the keys in the json. The values of the keys will be the input to the fill function of playwright.
                        - search: Search for the query. For this we will have the following keys in the json. 
                            - url: The url to navigate to.  The value of the key will be the input to the goto function of playwright.
                            - selector: Click on the element. The value of the key will be the input to the fill function of playwright.
                            - text: The text to type. The value of the key will be the input to the fill function of playwright.
                            - submit_selector: The selector for the login/submit button
                        - login: Perform login action. For this we will have the following keys in the json:
                            - url: The login page URL
                            - username_selector: The selector for the username/email field
                            - password_selector: The selector for the password field
                            - username: The username/email to enter
                            - password: The password to enter
                            - submit_selector: The selector for the login/submit button
                    - needs_page_info: A boolean indicating whether you need to see the current page state before proceeding. If the user command has been executed, set this to false. Only set this to true if the user command has not been executed, confirm this from the screenshot.
                    - extract_data: If needs_page_info is true, then you need to extract data from the page. Give a natural language description of the data you need to extract, this will be passed to another AI agent to extract the data based on the selector and attribute. This is required if needs_page_info is true.

                    Extra Information:
                    - If you are trying to input something at google.com then the selector is "textarea[name="q"]"
                        """
                                  )
                ]

                # Add the user's command
                messages.append(HumanMessage(content=command))

                # Add context from chat history (last 10 messages)
                for msg in self.chat_history[-10:]:
                    if msg.get("role") == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        messages.append(AIMessage(content=msg["content"]))

                # Add page information if available
                if page_info:
                    messages.append(
                        HumanMessage(
                            content=[
                                {
                                    "type": "text",
                                    "text": f"Current URL: {page_info['url']}\nCurrent Title: {page_info['title']}",
                                    "extract_data": page_info["data"]
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{page_info['screenshot']}",
                                        "detail": "low"
                                    }
                                }
                            ]
                        )
                    )

                # Create a structured output chain with LangChain
                chain = self.llm | self.command_parser

                try:
                    # Execute the chain and get structured output
                    parsed_command = chain.invoke(messages)

                    # Log the parsed command
                    logger.info(f"Parsed command: {parsed_command}")

                    # Store in chat history
                    self.chat_history.append({
                        "role": "assistant",
                        "content": json.dumps(parsed_command)
                    })

                except Exception as e:
                    logger.error(f"AI response parsing error: {str(e)}")
                    self.chat_history.append({
                        "role": "user",
                        "content": f"Failed to parse AI response: {str(e)}"
                    })
                    continue

                # Execute the parsed command
                try:
                    if parsed_command.get("action") == "navigate":
                        logger.info(f"Navigating to {parsed_command['url']}")
                        await self.page.goto(parsed_command["url"])
                    elif parsed_command.get("action") == "click":
                        logger.info(
                            f"Clicking element with selector: {parsed_command['selector']}")
                        await self.page.click(parsed_command["selector"])
                    elif parsed_command.get("action") == "type":
                        logger.info(
                            f"Typing '{parsed_command['text']}' into selector: {parsed_command['selector']}")
                        await self.page.fill(parsed_command["selector"], parsed_command["text"])
                    elif parsed_command.get("action") == "search":
                        logger.info(
                            f"Searching '{parsed_command['text']}' at {parsed_command['url']}")
                        await self.page.goto(parsed_command["url"])
                        await self.page.fill(parsed_command["selector"], parsed_command["text"])
                        await self.page.keyboard.press("Enter")
                    elif parsed_command.get("action") == "login":
                        logger.info(
                            f"Performing login at {parsed_command['url']}")
                        await self.page.goto(parsed_command["url"])
                        await self.page.fill(parsed_command["username_selector"], parsed_command["username"])
                        await self.page.fill(parsed_command["password_selector"], parsed_command["password"])
                        await self.page.click(parsed_command["submit_selector"])

                    await self.page.wait_for_load_state("domcontentloaded")
                    await self.page.wait_for_load_state("networkidle", timeout=10000)
                    # Additional wait to ensure JavaScript has completed
                    await self.page.wait_for_function("document.readyState === 'complete'", timeout=10000)
                except Exception as e:
                    logger.error(f"Command execution error: {str(e)}")
                    self.chat_history.append({
                        "role": "user",
                        "content": f"Failed to execute command: {str(e)}"
                    })
                    continue

                # Check if we need page information
                if parsed_command.get("needs_page_info", False):
                    logger.info("Getting page information as requested")
                    page_info = await self.get_page_info(parsed_command.get("extract_data"))
                else:
                    logger.info("Command execution complete")
                    run_agent = False

            return {
                "status": "success",
                "message": f"Executed command: {command}",
                "conversation_history": self.chat_history
            }

        except Exception as e:
            logger.error(f"General error in execute_command: {str(e)}")
            return {"status": "error", "message": str(e)}

    async def extract_data(self, selector: str, attribute: Optional[str] = None, multiple: bool = False):
        """
        Extract data from web elements using the specified selector.

        Args:
            selector: CSS selector to find elements
            attribute: Optional attribute name to extract. If None, extracts text content
            multiple: If True, returns data from all matching elements; if False, only from the first match

        Returns:
            Dict containing status, data extracted, and other relevant information
        """
        try:
            # Check if selector exists on the page
            elements = await self.page.query_selector_all(selector)

            if not elements:
                return {
                    "status": "error",
                    "message": f"No elements found matching selector: {selector}"
                }

            result_data = []

            # Process either all matching elements or just the first one
            target_elements = elements if multiple else [elements[0]]

            for element in target_elements:
                if attribute:
                    # Extract the specified attribute
                    value = await element.get_attribute(attribute)
                else:
                    # Extract text content if no attribute specified
                    value = await element.text_content()

                # Clean up extracted data
                if value:
                    value = value.strip()
                    result_data.append(value)

            return {
                "status": "success",
                "data": result_data if multiple else result_data[0] if result_data else None,
                "count": len(result_data),
                "selector": selector,
                "attribute": attribute
            }

        except Exception as e:
            print(f"Error extracting data: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to extract data: {str(e)}",
                "selector": selector,
                "attribute": attribute
            }

    async def extract(self, query: str) -> Dict[str, Any]:
        try:
            # Get current page information
            page_info = await self.get_page_info()

            # Prepare messages for OpenAI
            messages = [
                SystemMessage(content="""
                    You are a web scraping assistant. Your task is to determine the best CSS selector and attribute to extract data
                    from a webpage based on a natural language query. Your response should be parsable using Python's json.loads function.

                    The json should have the following fields:
                    - selector: The CSS selector to use
                    - attribute: The attribute to extract (optional, if not provided, text content will be extracted)
                    - multiple: Boolean indicating whether to get all matching elements or just the first one
                    - explanation: A brief explanation of why these parameters were chosen
                    """
                              ),
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                                    "text": f"Current URL: {page_info['url']}\nCurrent Title: {page_info['title']}",
                                    "extract_data": page_info["data"]
                        },
                        {
                            "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{page_info['screenshot']}",
                                        "detail": "low"
                                    }
                        }
                    ]
                )
            ]

            # Get response from OpenAI using direct client (for multimodal)
            logger.info(f"Processing extraction query: {query}")
            chain = self.llm | self.extract_parser

            # Parse the response
            try:
                extraction_params = chain.invoke(messages)
                logger.info(
                    f"Parsed extraction parameters: {extraction_params}")

                # Validate extraction params
                required_fields = ["selector", "explanation"]
                for field in required_fields:
                    if field not in extraction_params:
                        return {
                            "status": "error",
                            "message": f"Missing required field '{field}' in extraction parameters"
                        }

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI response: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Failed to parse AI response: {str(e)}"
                }

            # Extract data using the determined parameters
            logger.info(
                f"Extracting data with params: {extraction_params}")
            result = await self.extract_data(
                selector=extraction_params["selector"],
                attribute=extraction_params.get("attribute"),
                multiple=extraction_params.get("multiple", False)
            )

            # Add explanation to the result
            if result["status"] == "success":
                result["explanation"] = extraction_params["explanation"]

            # Log successful extraction
            data_preview = str(result["data"])[
                : 100] + "..." if len(str(result["data"])) > 100 else str(result["data"])
            logger.info(f"Successfully extracted data: {data_preview}")

            return result

        except Exception as e:
            logger.error(f"Error in extract: {str(e)}")
            return {"status": "error", "message": str(e)}
