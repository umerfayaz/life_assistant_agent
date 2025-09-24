from ai_sidekick.calendar_integration import create_event, update_event, list_events
from playwright.async_api import async_playwright
from langgraph.prebuilt import ToolNode
from typing import Optional
from .duffel_client import search_flights, book_flight
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field
import requests
import smtplib
from langchain_core.tools import StructuredTool
from langchain_core.tools import tool
from email.mime.text import MIMEText

from email.mime.multipart import MIMEMultipart
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
import asyncio

load_dotenv(override=True)

# # Global variables for API credentials
# pushover_token = os.getenv("PUSHOVER_TOKEN")
# pushover_user = os.getenv("PUSHOVER_USER")
# pushover_url = "https://api.pushover.net/1/messages.json"

# Initialize search wrapper with error handling
try:
    serper = GoogleSerperAPIWrapper(serper_api_key=os.getenv("SERPER_API_KEY"))
    print("✓ Serper API initialized successfully")
except Exception as e:
    print(f"⚠ Warning: Could not initialize Serper API: {e}")
    serper = None


async def playwright_tools():
    """
    Initialize Playwright browser toolkit with proper error handling
    """
    try:
        print(" Initializing Playwright browser...")
        playwright = await async_playwright().start()
        
        # Launch browser with better configuration
        browser = await playwright.chromium.launch(
            headless=False,  # Set to True for production
            args=[
                '--no-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-first-run',
                '--disable-default-apps'
            ]
        )
        
        # Create toolkit
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        tools = toolkit.get_tools()
        
        print(f"✓ Playwright initialized with {len(tools)} browser tools")
        return tools, browser, playwright
        
    except Exception as e:
        print(f"✗ Error initializing Playwright: {e}")
        print(" Returning empty tools list - browser functionality will be disabled")
        return [], None, None


class EmailInput(BaseModel):
    subject: str
    body: str
    to_email: str


def send_email(subject: str, body: str, to_email: str) -> str:
    """
    Send an email notification to the user
    """
    try:
        email_host = os.getenv("EMAIL_HOST")
        email_port = int(os.getenv("EMAIL_PORT", 587))
        email_user = os.getenv("EMAIL_USER")
        email_pass = os.getenv("EMAIL_PASS")

        if not email_host or not email_user or not email_pass:
            print("Email credentials missing in .env")
            return "Error: Email credentials are not configured"

        msg = MIMEMultipart()
        msg['From'] = email_user
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(email_host, email_port) as server:
            server.starttls()
            server.login(email_user, email_pass)
            server.send_message(msg)

        return "Email sent successfully"

    except Exception as e:
        return f"Email sending error: {str(e)}"

@tool
def slack_notification_tool(message: str) -> str:
    """Send a notification to Slack channel"""
    try:
        import os
        import requests
        
        # Get Slack webhook URL from environment
        slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
        
        if not slack_webhook_url:
            return "Slack webhook URL not configured"
        
        # Send to Slack
        payload = {
            "text": message,
            "username": "AI Assistant",
            "icon_emoji": ":robot_face:"
        }
        
        response = requests.post(slack_webhook_url, json=payload, timeout=10)
        
        if response.status_code == 200:
            return f"Slack notification sent successfully: {message[:50]}..."
        else:
            return f"Failed to send Slack notification: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Failed to send Slack notification: {str(e)}"


class FlightSearchInput(BaseModel):
    origin: str = Field(description= "Origin Airport IATA code (eg. , KHI) ")
    destination: str = Field(description = "Destination airport IATA code(eg. , LHE)")
    departure_date: str = Field(description = " Departure date in YYYY-MM-DD format")
    passengers: int = Field(description = "Number of passengers (Integer)", ge=1 , le= 9)

class config:
       validate_assigment = True

def search_flights_structured(origin: str, destination: str, departure_date: str, passengers: int = 1) -> str:

    try:


        if isinstance(passengers, str):
            passengers =int(passengers)

        flight_perams ={
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "passengers": passengers

        }

        print(f" Search Flights {flight_perams}")
        return search_flights(**flight_perams)
    except ValueError as e:
        return f"Flight search error: No passengers added {str(e)}"
    

    

class BookFlightInput(BaseModel):
    offer_id: str = Field(description = "Duffel flight Offer ID")
    passenger_name: str = Field(description = "Passenger full name")
    email: str = Field(description = "Passenger email address")


def book_flight_structured(offer_id: str, passenger_name: str, email: str) -> str:

    try:
        booking_perams ={
            "offer_id": offer_id,
            "passenger_name": passenger_name,
            "email": email
        }
        return book_flight(**booking_perams)
    except Exception as e:
        print(f"Flight booking error{str(e)}")


class BookMeetingInput(BaseModel):
    summary: str = Field(description = "Meeting title/name")
    description: str = Field(default = "", description = "Meeting description")
    start_time: str = Field(description = "Start time in ISO format (YYY-DDTHH:MM:SS)")
    end_time: str = Field(description = "End time in ISO format (YYYY-DDTHH:MM:SS)")

class GetMeetingInput(BaseModel):
    n: int = Field(default=5, description = "Number of meetings retrieve")

class RescheduleMeetingInput(BaseModel):
    event_id: Optional[str] = Field(default =None, description = " Google Calendar Event ID to update")
    summary: str = Field(default = None, description = "New Meeting Title")
    description: str = Field(default=None, description = "New Meeting description")
    start_time: str = Field(default = None, description = "New start time in ISO format")
    end_time: str = Field(default=None, description = "New end time is ISO format")


def book_meeting_structured(summary: str, description: str ="", start_time: str = None, end_time: str = None) -> str:

    try:  
        print(f"Book meeting_structured called with: {summary}, {start_time}, {end_time}")

        if not summary.strip():
            return  "Error: Meeting summaris is required"
        
        if not start_time or not end_time:
            return "Both Start_time and end_time required"
        
        result = create_event (
            summary = summary,
            description = description,
            start_time = start_time,
            end_time = end_time
        )

        print(f" create event result {result}")

        return result 

    except Exception as e:
        error_msg = f"Error book meeting{str(e)}"
        print(f"Eceptopn: {error_msg}")
        return error_msg


def get_structured_meeting(n: int = 5,) -> str:

    try:

      print(f"Get metting_structured with n={n}")

      result = list_events(n)

      print(f"List event results {result[:100]}...")
      return result

    except Exception as e:
        error_msg =(f"Error: Meeting Structured {str(e)}")
        print(f"Exceoption {error_msg}")
        return error_msg

def reschedule_meeting_structured(event_id: Optional[str] = None, summary: Optional[str] = None, description: Optional[str] = None, start_time: Optional[str] = None, end_time: Optional[str] = None) -> str:
    try:
        print(f"[DEBUG] Reschedule called with -> event_id={event_id}, summary={summary}, start_time={start_time}, end_time={end_time}")

        # If no event_id provided, try to find it
        if not event_id:
            if summary:
                try:
                    print(f"[DEBUG] Looking for event with summary: '{summary}'")
                    meetings_result = list_events(20)  # Get recent meetings
                    print(f"[DEBUG] Raw meetings result: {meetings_result[:200]}...")  # First 200 chars
                    
                    # Handle different response formats
                    events_to_search = []
                    
                    if isinstance(meetings_result, str):
                        try:
                            import json
                            meetings_data = json.loads(meetings_result)
                            if isinstance(meetings_data, dict) and 'items' in meetings_data:
                                events_to_search = meetings_data['items']
                            elif isinstance(meetings_data, list):
                                events_to_search = meetings_data
                        except json.JSONDecodeError:
                            # Maybe it's a formatted string, try to parse it differently
                            print(f"[DEBUG] Could not parse as JSON, treating as string")
                            # You might need to implement string parsing here based on your list_events format
                            return f"Error: Could not parse calendar events. Raw response: {meetings_result[:100]}..."
                    elif isinstance(meetings_result, dict):
                        if 'items' in meetings_result:
                            events_to_search = meetings_result['items']
                        else:
                            events_to_search = [meetings_result]
                    elif isinstance(meetings_result, list):
                        events_to_search = meetings_result
                    
                    print(f"[DEBUG] Found {len(events_to_search)} events to search")
                    
                    # Look for matching event by summary (case-insensitive)
                    found_event_id = None
                    for event in events_to_search:
                        if isinstance(event, dict):
                            event_summary = event.get('summary', '') or event.get('title', '')
                            event_id_candidate = event.get('id', '') or event.get('event_id', '')
                            
                            print(f"[DEBUG] Checking event: '{event_summary}' (ID: {event_id_candidate})")
                            
                            # Try exact match first
                            if event_summary.lower().strip() == summary.lower().strip():
                                found_event_id = event_id_candidate
                                print(f"[DEBUG] Exact match found! Event ID: {found_event_id}")
                                break
                            
                            # Try partial match
                            elif summary.lower().strip() in event_summary.lower() or event_summary.lower() in summary.lower().strip():
                                found_event_id = event_id_candidate
                                print(f"[DEBUG] Partial match found! Event ID: {found_event_id}")
                                # Don't break here, continue looking for exact match
                    
                    if found_event_id:
                        event_id = found_event_id
                        print(f"[DEBUG] Using event ID: {event_id}")
                    else:
                        # List all available events for debugging
                        available_events = []
                        for event in events_to_search:
                            event_summary = event.get('summary', '')
                            if event_summary:
                                available_events.append(event_summary)
                        
                        return f"Error: Could not find a meeting with title '{summary}'. Available meetings: {', '.join(available_events[:5])}{'...' if len(available_events) > 5 else ''}"
                        
                except Exception as lookup_error:
                    print(f"[DEBUG] Error during event lookup: {lookup_error}")
                    import traceback
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                    return f"Error: Could not search for existing meetings: {str(lookup_error)}"
            else:
                return "Error: Either event_id or meeting summary is required for rescheduling."

        # Validate that we have something to update
        if not any([summary, description, start_time, end_time]):
            return "Error: At least one field (summary, description, start_time, or end_time) must be provided for rescheduling."

        print(f"[DEBUG] Calling update_event with event_id={event_id}")
        result = update_event(
            event_id=event_id,
            summary=summary,
            description=description,
            start_time=start_time,
            end_time=end_time
        )

        print(f"[DEBUG] Update result: {result}")
        return result

    except Exception as e:
        error_msg = f"Reschedule meeting failed: {str(e)}"
        print(f"[DEBUG] Exception in reschedule: {error_msg}")
        import traceback
        print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
        return error_msg


 

def get_file_tools():
    """
    Get file management tools with error handling
    """
    try:
        # Ensure sandbox directory exists
        sandbox_dir = "sandbox"
        os.makedirs(sandbox_dir, exist_ok=True)
        
        toolkit = FileManagementToolkit(root_dir=sandbox_dir)
        tools = toolkit.get_tools()
        
        print(f"✓ File management initialized with {len(tools)} tools")
        return tools
        
    except Exception as e:
        print(f"⚠ Warning: Could not initialize file tools: {e}")
        return []


def safe_search(query: str) -> str:
    """
    Safe search function with error handling
    """
    try:
        if not serper:
            return "Search unavailable: Please configure SERPER_API_KEY in your .env file"
        
        if not query or not query.strip():
            return "Error: Empty search query"
            
        print(f" Searching for: {query}")  # Debug log
        result = serper.run(query.strip())
        
        if result and len(result.strip()) > 0:
            print(f" Search successful: {len(result)} characters returned")
            return result
        else:
            return f"No search results found for: {query}"
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return f"Search error: {str(e)}"


def safe_wikipedia_search(query: str) -> str:
    """
    Safe Wikipedia search with error handling
    """
    try:
        if not query or not query.strip():
            return "Error: Empty Wikipedia query"
            
        wikipedia = WikipediaAPIWrapper()
        wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)
        result = wiki_tool.run(query.strip())
        return result if result else "No Wikipedia results found"
        
    except Exception as e:
        return f"Wikipedia search error: {str(e)}"

def safe_python_exec(code: str) -> str:
    """
    Safe Python execution with error handling
    """
    try:
        if not code or not code.strip():
            return "Error: Empty code provided"
            
        python_repl = PythonREPLTool()
        result = python_repl.run(code.strip())
        return result if result else "Code executed but no output returned"
        
    except Exception as e:
        return f"Python execution error: {str(e)}"


async def other_tools():
    """
    Get additional tools with comprehensive error handling
    """
    tools_list = []

    # Flight search tool
    try:
        flight_tool = StructuredTool.from_function(
            name="Flight_Search",
            func=search_flights_structured,
            args_schema=FlightSearchInput,
            description=( "Search for flights between airports. Parameters:\n"
            "- origin: IATA airport code (string, e.g. 'KHI')\n"
            "- destination: IATA airport code (string, e.g. 'DXB')\n" 
            "- departure_date: Date in YYYY-MM-DD format (string, e.g. '2025-08-27')\n"
            "- passengers: Number of passengers (integer, e.g. 1, not '1')"
            )
        )
        tools_list.append(flight_tool)
        print("✓ Flight search tool added")
    except Exception as e:
        print(f"⚠ Flight search tool error: {e}")

    # Email tool
    try:
        email_tool = StructuredTool.from_function(
            name="reply_email",
            func=send_email,
            args_schema=EmailInput,
            description=(
                "Send an email. Fields: subject (str), body (str), to_email (str)."
            )
        )
        tools_list.append(email_tool)
        print("✓ Email tool added")
    except Exception as e:
        print(f"⚠ Email tool error: {e}")

    # Flight booking tool
    try:
        booking_tool = StructuredTool.from_function(
            name="book_flight",
            func=book_flight_structured,
            args_schema= BookFlightInput,
            description=("Book a flight using Duffer offer ID. Requires Offer_id, passenger_name and email"
            )
        )
        tools_list.append(booking_tool)
        print("✓ Flight booking tool added")
    except Exception as e:
        print(f"⚠ Flight booking tool error: {e}")
   

    # File management tools
    try:
        file_tools = get_file_tools()
        tools_list.extend(file_tools)
    except Exception as e:
        print(f"⚠ File tools error: {e}")

    # Web search tool
    try:
        if serper:
            search_tool = Tool(
                name="web_search",
                func=safe_search,
                description="Search the web for current information including weather, news, facts, and real-time data. Use this when you need up-to-date information that you don't have in your training data. Example: 'weather in Karachi', 'latest news', 'current stock prices'."
            )
            tools_list.append(search_tool)
            print("✓ Web search tool added")
        else:
            print("⚠ Search tool skipped - SERPER_API_KEY not configured")
    except Exception as e:
        print(f"⚠ Web search tool error: {e}")

    # Wikipedia tool
    try:
        wikipedia_tool = Tool(
            name="wikipedia_search",
            func=safe_wikipedia_search,
            description="Search Wikipedia for encyclopedic information. Use this for factual information about topics, people, places, etc."
        )
        tools_list.append(wikipedia_tool)
        print("✓ Wikipedia tool added")
    except Exception as e:
        print(f"⚠ Wikipedia tool error: {e}")

    # Python REPL tool
    try:
        python_tool = Tool(
            name="python_repl",
            func=safe_python_exec,
            description="Execute Python code. Use this for calculations, data processing, or when you need to run code. Always include print() statements to see output."
        )
        tools_list.append(python_tool)
        print("✓ Python REPL tool added")
    except Exception as e:
        print(f"⚠ Python tool error: {e}")

    try:
        calendar_book_tool = StructuredTool.from_function(
            name="booking_meeting",
            func=book_meeting_structured,
            args_schema  = BookMeetingInput,
            description="Book a meeting in Google Calendar. requires start_time and end_time in ISO format (YYYY-MM-DDTHH:MM:SS)" 

        )
        tools_list.append(calendar_book_tool)

        calendar_list_tool = StructuredTool.from_function(
            name="get_meetings",
            func=get_structured_meeting,
            args_schema = GetMeetingInput,
            description="List upcoming Google Calendar meetings. Optionally Specify number of meetings to retrieve"
        )
        tools_list.append(calendar_list_tool)

        calendar_update_tool = StructuredTool.from_function(
            name="reschedule_meeting",
            func=reschedule_meeting_structured,
            args_schema = RescheduleMeetingInput,
            description="Reschedule/update a meeting in Google Calendar. using the event ID"
        )
        tools_list.append(calendar_update_tool)
        print("✓ Google Calendar tools added with improved wrappers")
    except Exception as e:
        print(f"⚠ Calendar tools error: {e}")
        import traceback
        print(f"🔧 DEBUG: Calendar tools traceback: {traceback.format_exc()}")

    return tools_list

# Test function for debugging
async def test_tools():
    """
    Test function to verify all tools work correctly
    """
    print(" Testing tools...")
    
    try:
        # Test playwright tools
        playwright_tools_list, browser, playwright_instance = await playwright_tools()
        print(f"Playwright tools: {len(playwright_tools_list)}")
        
        # Test other tools
        other_tools_list = await other_tools()
        print(f"Other tools: {len(other_tools_list)}")
        
        # Clean up browser if created
        if browser:
            await browser.close()
        if playwright_instance:
            await playwright_instance.stop()
            
        print(" Tool testing completed successfully")
        return True
        
    except Exception as e:
        print(f" Tool testing failed: {e}")
        return False


# Run test if this file is executed directly
if __name__ == "__main__":
    asyncio.run(test_tools())

