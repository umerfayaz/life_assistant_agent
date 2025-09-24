import pickle
import os
from datetime import datetime, timedelta, tzinfo
import zoneinfo
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from dateutil import parser
from zoneinfo import ZoneInfo
import re
import dateparser

CREDENTIAL_FILE = "client_secret.json"

# ✅ Support both Calendar and Gmail
SCOPES = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",   # read emails
    "https://www.googleapis.com/auth/gmail.send",       # send emails
    "https://www.googleapis.com/auth/gmail.modify"      # mark as read etc.
]

KARACHI_TZ = ZoneInfo("Asia/Karachi")

def get_credentials():
    creds = None
    
    # Load existing token
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    
    # If no valid creds → login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"Error refreshing token: {e}")
                creds = None
        
        if not creds:
            if not os.path.exists(CREDENTIAL_FILE):
                raise FileNotFoundError(f"{CREDENTIAL_FILE} not found. Please download it from Google Cloud Console.")
            
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIAL_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
            
            # Save creds
            with open("token.pickle", "wb") as token:
                pickle.dump(creds, token)
    
    return creds


def get_calendar_service():
    creds = get_credentials()
    return build("calendar", "v3", credentials=creds)


def get_gmail_service():
    creds = get_credentials()
    return build("gmail", "v1", credentials=creds)


def fix_timezone(dt_str):
    """Fixed timezone handling function"""
    if not dt_str:
        return None
    
    print(f"DEBUG: Processing datetime string: {dt_str}")
    
    try:
        dt = parser.isoparse(dt_str)  # Parse ISO string
        print(f"DEBUG: Parsed datetime: {dt} (has timezone: {dt.tzinfo is not None})")
        
        if dt.tzinfo is None:  # If no timezone info, assume Karachi
            dt = dt.replace(tzinfo=KARACHI_TZ)
            print(f"DEBUG: Added Karachi timezone: {dt}")
        else:
            # Fixed syntax - remove 'tzinfo=' parameter
            dt = dt.astimezone(KARACHI_TZ)
            print(f"DEBUG: Converted to Karachi timezone: {dt}")
        
        result = dt.isoformat()
        print(f"DEBUG: Final datetime: {result}")
        return result
        
    except Exception as e:
        print(f"ERROR: Timezone conversion failed for '{dt_str}': {e}")
        # Fallback: assume it's already in Karachi time
        return dt_str


def create_event(summary, start_time, end_time, description="", location="", attendees=None):
    """Create a new calendar event"""
    try:
        service = get_calendar_service()
        if not service:
            return None
        
        if attendees is None:
            attendees = []
        
        start_time = fix_timezone(start_time)
        end_time = fix_timezone(end_time)
        
        print(f"DEBUG: Creating event with times - Start: {start_time}, End: {end_time}")

        event = {
            'summary': summary,
            'description': description,
            'location': location,
            'start': {
                'dateTime': start_time,
                'timeZone': 'Asia/Karachi'
            },
            'end': {
                'dateTime': end_time,
                'timeZone': 'Asia/Karachi'
            },
            'attendees': [{'email': email} for email in attendees],
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 60},
                    {'method': 'popup', 'minutes': 15},
                ],
            },
        }
        
        event_result = service.events().insert(calendarId="primary", body=event).execute()
        print(f"Event created:{event_result.get('htmlLink')}")
        return event_result

    except Exception as e:
        print(f"[Error] Failed to create event{e}")
        return None

def update_event(event_id, summary=None, description=None, start_time=None, end_time=None):
    """Update an existing calendar event"""
    try:
        service = get_calendar_service()
        
        if not event_id:
            return "Error: Event ID is required"
        
        # Get the existing event first
        try:
            existing_event = service.events().get(calendarId="primary", eventId=event_id).execute()
        except Exception as e:
            return f"Error: Event with ID '{event_id}' not found: {str(e)}"
        
        # Update fields if provided
        if summary:
            existing_event['summary'] = summary
        
        if description:
            existing_event['description'] = description
        
        if start_time:
            existing_event['start'] = {
                'dateTime': start_time,
                'timeZone': 'Asia/Karachi',
            }
        
        if end_time:
            existing_event['end'] = {
                'dateTime': end_time,
                'timeZone': 'Asia/Karachi',
            }
        
        # Update the event
        updated_event = service.events().update(
            calendarId="primary", 
            eventId=event_id, 
            body=existing_event
        ).execute()
        
        return f"✅ Event updated successfully!\n🔗 Link: {updated_event.get('htmlLink', 'No link available')}"
        
    except Exception as e:
        return f"Error updating event: {str(e)}"


def list_events(n=5):
    """List upcoming calendar events"""
    try:
        service = get_calendar_service()
        now = datetime.now().isoformat() + "Z"
        
        events_result = service.events().list(
            calendarId="primary", 
            timeMin=now, 
            maxResults=n, 
            singleEvents=True, 
            orderBy="startTime"
        ).execute()
        
        events = events_result.get("items", [])
        
        event_list = []
        for event in events:
            summary = event.get('summary', 'No title')
            start = event['start'].get('dateTime', event['start'].get('date'))
            event_id = event.get('id', 'No ID')
            
            # Format the datetime for better readability
            event_list.append({
                "start": start,
                "summary": summary,
                "id": event_id
            })

        return event_list
    except Exception as e:
        print(f"Error listing events: {e}")
        return []


def parse_user_instruction(instruction: str):
    """
    Parse a natural language user instruction into event details.
    Example: 'Book a meeting with Alex tomorrow at 12 PM for 90 minutes'
    """
    # 1. Extract event summary
    summary_match = re.search(r"(meeting|call|event|appointment|session|chat) with (.+?)(?:\s+tomorrow|\s+today|\s+next|\s+at|\s+on|$)", instruction, re.IGNORECASE)
    if summary_match:
        summary = f"{summary_match.group(1).capitalize()} with {summary_match.group(2).strip()}"
    else:
        summary = "Untitled Event"

    # 2. Extract duration (default = 60 minutes)
    duration_match = re.search(r"for (\d+)\s*(minutes|min|hours|hrs|h)?", instruction, re.IGNORECASE)
    duration = 60  # default 1 hour
    if duration_match:
        amount = int(duration_match.group(1))
        unit = duration_match.group(2) or "minutes"
        if "hour" in unit or "h" in unit:
            duration = amount * 60
        else:
            duration = amount

    # 3. Parse datetime
    now = datetime.now(KARACHI_TZ)
    dt = dateparser.parse(
        instruction,
        settings={
            "TIMEZONE": "Asia/Karachi",
            "RETURN_TIMEZONE_AS_AWARE": True,
            "PREFER_DATES_FROM_FUTURE": True,
            "RELATIVE_BASE": now
        },
    )
    if not dt:
        raise ValueError(f"Could not parse datetime from instructions: '{instruction}'")

    start_time = dt.astimezone(KARACHI_TZ)
    end_time = start_time + timedelta(minutes=duration)

    return summary, start_time.isoformat(), end_time.isoformat()


def create_event_from_instruction(instruction, description=""):
    """
    Fully autonomous event creator.
    Takes plain user instruction and creates Google Calendar event.
    """
    try:
        summary, start_time, end_time = parse_user_instruction(instruction)

        result = create_event(
            summary=summary,
            description=description,
            start_time=start_time,
            end_time=end_time
        )

        return f"✅ Scheduled: {summary}\n🕒 {start_time} → {end_time}\n{result}"

    except Exception as e:
        return f"⚠️ Could not schedule event: {e}"


def test_calendar():
    """Test function for Google Calendar integration"""
    try:
        print("🧪 Testing Google Calendar integration...")
        
        # Test creating an event
        result = create_event(
            summary="Project delay Meeting With alex",
            description="This is a test event from Mindara agent",
            start_time="2025-08-27T10:00:00",
            end_time="2025-08-27T11:00:00"
        )
        print(f"Create result: {result}")
        
        # Test listing events
        events = list_events(5)
        print(f"List result: {events}")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    test_calendar()