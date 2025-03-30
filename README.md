# AI Browser Automation Agent

An AI-powered browser automation agent that can control your browser using natural language commands.

## Features

- Natural language browser control
- Web page interaction automation

## Setup

1. Create a conda environment
```bash
conda create -n browser_automation --file=environments.yml python=3.13.2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Playwright browsers:
```bash
playwright install
```

4. Set up environment variables:
Create a `.env` file with:
```
OPENAI_API_KEY=your_api_key_here
```

5. Run the server:
```bash
uvicorn main:app --reload
```

## API Endpoints

### Interact API
- POST `/api/interact`
  - Accepts natural language commands
  - Runs the automation

## Example Usage

```python
import requests

# Example interaction
response = requests.post("http://localhost:8000/api/interact", 
    json={"command": "Go to google.com and search for 'aniket sharma' and open his LinkedIn page"})
print(response.json())
```
