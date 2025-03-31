# AI Browser Automation Agent

An AI-powered browser automation agent that can control your browser using natural language commands. The system includes a chat interface for interacting with the AI agent and managing browser automation tasks.

## Features

- **Interactive Chat Interface**: Real-time communication with the AI agent through a web-based chat interface
- **Browser Automation**: Execute browser commands and extract data from web pages
- **Chat History Management**: Persistent storage of chat history using MongoDB
- **Process Repetition**: Ability to repeat previous chat processes with a new agent
- **WebSocket Communication**: Real-time updates and responses through WebSocket connections

## Prerequisites

- Python
- MongoDB
- OpenAI API key
- Anaconda (for Python environment)
## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd browser_automation
```

2. Create conda environment and install dependencies:
```bash
conda env create -n browser_automation --file=environments.yml python=3.13.2
```
If the above fails because of version issues:
```bash
conda create -n browser_automation python=3.13.2
conda install openai
conda install fastapi
pip install playwright 
pip install playwright-stealth
conda install uvicorn 
pip install dotenv
conda install requests
conda install websockets
conda install pymongo
```

3. Install Playwright browsers:
```bash
playwright install
```

4. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
MONGODB_URL=mongodb://localhost:27017
```

## Usage

### Running the app

1. Start the application:
```bash
python main.py
```

2. Open your browser and navigate to `http://localhost:8000`

3. Start a new chat or continue an existing one

4. Interact with the AI agent using natural language commands

### Interact API

```python
import requests

# Example interaction
response = requests.post("http://localhost:8000/api/interact", 
    json={"command": "Go to google.com and search for 'aniket sharma' and open his LinkedIn page"})
print(response.json())
```

### Extract API

```python
import requests

response = requests.post("http://localhost:8000/api/interact", 
    json={"command": "Go to google.com and search for 'aniket sharma'"})
# Example extraction
response = requests.post("http://localhost:8000/api/extract", 
    json={"query": "Get all links from the page"})
print(response.json())
```

### Process Repetition

To repeat a previous chat process:

1. Click the "Repeat Process" button
2. The system will:
   - Create a new chat
   - Process all user messages from the original chat
   - Show both user messages and assistant responses in real-time
   - Maintain the same sequence of commands and interactions


### Automated Process Repetition

To automatically repeat the process at regular intervals using cron:

1. Make sure the main application is running:
```bash
python main.py
```

2. Set up a cron job to run the repeat process script with your chat ID:
```bash
# Open crontab editor
crontab -e

# Add the following line to run every 5 minutes
# Replace YOUR_CHAT_ID with the actual chat ID you want to repeat
*/5 * * * * cd /path/to/browser_automation && /path/to/python repeat_process.py --chat_id YOUR_CHAT_ID >> /path/to/browser-automation/repeat_process.log 2>&1
```

3. Make the script executable:
```bash
chmod +x repeat_process.py
```

The cron job will:
- Run the repeat process at the specified interval
- Log output to `repeat_process.log`
- Create a new agent for each run
- Process all messages from the original chat
- Show messages in real-time through the web interface using WebSocket communication

You can also run the script manually:
```bash
python repeat_process.py --chat_id YOUR_CHAT_ID
```

Note: The main application must be running for the WebSocket connection to work and show messages interactively.

## Architecture

- **Frontend**: HTML/JavaScript
- **Backend**: FastAPI
- **Database**: MongoDB
- **Browser Agent**: Playwright

## API Endpoints

- `POST /api/interact`
  - Accepts natural language commands
  - Runs the automation
- `POST /api/extract`
  - Accepts natural language commands
  - Extracts the requested data and returns as a JSON object
- `POST /api/chat/create`: Create a new chat
- `GET /api/chat/{chat_id}/history`: Get chat history
- `POST /api/chat/{chat_id}/message`: Send a message to the chat
- `POST /api/chat/{chat_id}/repeat`: Repeat the chat process
- `POST /api/chat/{chat_id}/process_next_message`: Process the next message in a repeated process
- `WebSocket /api/ws/chat/{chat_id}`: WebSocket endpoint for real-time communication

## Error Handling

The system includes comprehensive error handling for:
- Database connection issues
- WebSocket disconnections
- Browser automation errors
- API request failures
- Invalid message formats
