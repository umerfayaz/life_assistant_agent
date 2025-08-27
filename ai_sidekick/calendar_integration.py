import pickle
import os
from datetime import datetime, tzinfo
import zoneinfo
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from dateutil import parser
from zoneinfo import ZoneInfo

CREDENTIAL_FILE = "client_secret.json"

scopes = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    creds = None
    
    # Load existing token
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    
    # If there are no (valid) credentials available, let the user log in.
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
            
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIAL_FILE, scopes
            )
            creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open("token.pickle", "wb") as token:
                pickle.dump(creds, token)
    
    service = build("calendar", "v3", credentials=creds)
    return service

def create_event(summary, description="", start_time=None, end_time=None):
    try:
        service = get_calendar_service()
        
        if not summary:
            return "Error: Event summary is required"
        
        KARACHI_TZ = ZoneInfo("Asia/Karachi")
        
        # Fixed timezone handling function
        def fix_timezone(dt_str):
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

        start_time = fix_timezone(start_time)
        end_time = fix_timezone(end_time)
        
        print(f"DEBUG: Creating event with times - Start: {start_time}, End: {end_time}")

        event = {
            "summary": summary,
            "description": description or "",
            "start": {
                "dateTime": start_time,
                "timeZone": "Asia/Karachi"
            },
            "end": {
                "dateTime": end_time,
                "timeZone": "Asia/Karachi"
            },
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "email", "minutes": 24 * 60},
                    {"method": "popup", "minutes": 10},
                ],
            },
        }
        
        created_event = service.events().insert(calendarId="primary", body=event).execute()
        
        event_link = created_event.get('htmlLink', 'No link available')
        event_id = created_event.get('id', 'No ID available')
        
        return f"Event '{summary}' created successfully!\nEvent ID: {event_id}\nLink: {event_link}"
        
    except Exception as e:
        print(f"Exception in create_event: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return f"Error creating event: {str(e)}"

def update_event(event_id, summary=None, description=None, start_time=None, end_time=None):
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
        
        if not events:
            return "No upcoming events found."
        
        event_list = []
        for event in events:
            summary = event.get('summary', 'No title')
            start = event['start'].get('dateTime', event['start'].get('date'))
            event_id = event.get('id', 'No ID')
            
            # Format the datetime for better readability
            try:
                if 'T' in start:
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    formatted_time = start_dt.strftime('%Y-%m-%d %H:%M')
                else:
                    formatted_time = start
            except:
                formatted_time = start
            
            event_list.append(f"📅 {summary}\n   🕐 {formatted_time}\n   🆔 {event_id}")
        
        return f"📋 Upcoming events:\n\n" + "\n\n".join(event_list)
        
    except Exception as e:
        return f"Error listing events: {str(e)}"

# Test function
def test_calendar():
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



