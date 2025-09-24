import asyncio
import os
import re
import json
from datetime import datetime, timedelta, timezone
from dateutil import parser
from langchain_core.messages import HumanMessage, SystemMessage
import requests
import logging
from ai_sidekick.sidekick import Sidekick
from ai_sidekick.sidekick_tools import slack_notification_tool 
from ai_sidekick.calendar_integration import list_events, get_gmail_service


# Basic logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_meeting_request(email_content, email_subject):
    full_txt = f"{email_content, email_subject}"

    meeting_keywords = [

        "meeting", "appointment", "schedule", "sync", "catch up", "update", 
        "call", "discussion", "conference", "zoom", "teams", "interview", 
        "presentation", "demo", "review", "planning", "standup", "scrum",
        "coffee", "lunch", "dinner", "meet", "gather", "session", "workshop"
    ]


    time_keywords = [

       "tomorrow", "today", "monday", "tuesday", "wednesday", "thursday", 
        "friday", "saturday", "sunday", "am", "pm", "next week", "next month",
        "at", "on", "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", 
        "oct", "nov", "dec", "time", "clock", "hour", "minute"
    ]


    has_meeting = any(keyword in full_txt for keyword in meeting_keywords)
    has_time = any(keyword in full_txt for keyword in time_keywords)

    print(f"[Debug] Meeting detecttion content: {full_txt [:100]}")
    print(f"[Debug] Meeting keywords has found: {meeting_keywords}")
    print(f"[Debug] Time keywords has found: {time_keywords}")

    return has_meeting and has_time


def extract_simple_meeting_info(email_content, email_subject, sender_email):
    """Extract basic meeting info from simple phrases like 'tomorrow 3pm'."""

    full_text = f"{email_content} {email_subject} {sender_email}".lower()

    meeting_info = {
        'title': email_subject if email_subject else "Meeting Request",
        'start_time': None,
        'end_time': None,
        'description': (
            f"Meeting scheduled from email\n\nFrom: {sender_email}\n"
            f"Original: {email_content[:100]}..."
        ),
        'attendees': [sender_email] if sender_email else []
    }

    import re
    from dateutil import parser as date_parser

    potential_times= re.findall(

        r'(?:tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
        r'next week|next month|'
        r'\d{1,2}[:/]\d{1,2}(?:[:/]\d{2,4})?|'
        r'\d{1,2}\s*(?:am|pm)|'
        r'\d{1,2}:\d{2}\s*(?:am|pm)?|'
        r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*|'
        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}|'
        r'\d{1,2}/\d{1,2}(?:/\d{2,4})?)', 
        full_text, re.IGNORECASE
    )


    print(f"[Debug] Found potential time expressions: {potential_times}")

    for time_expr in potential_times:
        try:
            parsed_time = date_parser.parse(time_expr, fuzzy= True, default=datetime.now())

            if parsed_time <datetime.now():
                if 'tomorrow' in time_expr.lower():
                    parsed_time += timedelta(days=1)
                elif any(day in time_expr.lower() for day in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']):

                    days_ahead =(parsed_time.weekday() - datetime.now().weekday()) % 7

                    if days_ahead < 0:
                        days_ahead += 7
                    parsed_time = datetime.now().replace(hour=parsed_time.hour, minute=parsed_time.minute) + timedelta(days=days_ahead)
                else:
                    parsed_time = parsed_time.replace(year=parsed_time.year +1)
            
            meeting_info['start_time'] = parsed_time.isoformat()
            meeting_info['end_time'] = (parsed_time + timedelta(hours=1)).isoformat()

            print(f"[Debug] Successfully parsed time: {parsed_time}")
            break

        except (ValueError, TypeError) as e:
            print(f"[Debug] Could not parse {time_expr}: {e}")
    
    if not meeting_info['start_time']:
        print(f"[Debug] No valid time pattern found in {full_text}")

    return meeting_info 


async def extract_meeting_with_llm(email_content, email_subject, sender_email, agent):
    """ Use LLM to extract meeting details when the pattern matching fails"""

    prompt = f"""

    Extract Meeting details from this email:

    subject: {email_subject}
    content: {email_content}
    From: {sender_email}

If this email contains a meeting request with date/time information, respond with JSON:
{{"has_meeting": true, "date": "YYYY-MM-DD", "time": "HH:MM", "title": "meeting title"}}

If no clear meeting details found, respond with:
{{"has_meeting": false}}

Extract any date/time mentioned, even if informal like "tomorrow 3pm" or "next Tuesday".
"""
    try:
        response = await agent.worker_llm_with_tools.ainvoke([
            SystemMessage(content= "Extarct meeting details from emails. Response with JSON only"),
            HumanMessage(content=prompt)
        ])

        result = json.loads(response.content.strip())

        if result.get("has_meeting"):

            date_str = result.get("date")
            time_str = result.get("time")

            if date_str and time_str:
                meeting_datetime = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
                return {
                    'title': result.get("title", email_subject),
                    'start_time': meeting_datetime.isoformat(),
                    'end_time': (meeting_datetime + timedelta(hours=1)).isoformat(),
                    'description': f"Meeting Scheduled from\n\nFrom: {sender_email}\nOriginal: {email_content[:100]}...",
                    'attendees': [sender_email] if sender_email else []
                }
    except Exception as e:
            print(f"[Debug] Extractionfailed {e}")
    return None

async def book_meeting_if_detected(emails_list):
    """ Book meeting if Detected"""
    print(f"[Debug] book_meeting_if_detected called with {len(emails_list)} emails")

    for email in emails_list:
        if detect_meeting_request(email.get('snippet',''), email.get('subject', '')):
            print(f"[Debug]  Meeting detected from{email.get('from')}")

            meeting_info = extract_simple_meeting_info(
                email.get('snippet', ''),
                email.get('subject', ''),
                email.get('from', '')
            )

            if not meeting_info or not meeting_info.get('start_time'):
                print(f"[Debug] Falling back to the llm for extracttion for {email.get('from')}")

                meeting_info = await extract_meeting_with_llm(
                    email.get('snippet', '' ),
                    email.get('subject', ''),
                    email.get('from', '')
                )

            if meeting_info and meeting_info.get('start_time'):
                print(f"Valid meeting info found, attempting too book")
                try:
                    from ai_sidekick.calendar_integration import create_event

                    event_result = create_event(
                        summary=meeting_info['title'],
                        start_time=meeting_info['start_time'],
                        end_time=meeting_info['end_time'],
                        description=meeting_info['description'],
                        attendees=meeting_info['attendees']
                    )
                    if event_result:
                        start_time = parser.parse(meeting_info['start_time'])
                        formatted_time = start_time.strftime("%A, %B %d at %I:%M %p")

                        await slack_notification_async(f"Meeting booke: {meeting_info['title']} on {formatted_time} with {email.get('from', '')}")
                        print(f"Meeting Successfully Booked: {meeting_info['title']}")

                        return f"I,ve Booked meeting successfully on {formatted_time}. A calendar invitiation hase been sent"

                except Exception as e:
                    print(f"Failed to book meeting {e}")
                    import traceback
                    traceback.print_exc()

                    await slack_notification_async(f"Failed to book the meeting from: {email.get('from', '')}:{str(e)}")
                    return None
            
            else:
                print(f"[Debug] Meeting detected but no valid time found")
                await slack_notification_async(f"Meeting detected from {email.get('from', '')} but couldn't extract complete details - Manual review required")
                return None
    return None

def slack_notification(message: str) -> str:
    """Send notification to Slack using the tool synchronously"""
    try:
        result = slack_notification_tool.invoke({"message": message})
        return result
    except Exception as e:
        print(f"Slack notification error: {e}")
        return f"Failed: {str(e)}"

async def slack_notification_async(message: str) -> str:
    """Send notification to Slack using the tool asynchronously"""
    try:
        result = await slack_notification_tool.ainvoke({"message": message})
        return result
    except Exception as e:
        print(f"Slack notification error: {e}")
        return f"Failed: {str(e)}"

def extract_email_decision(response):
    """Extract email decision with better error handling"""
    try:
        if isinstance(response, list):
            for msg in response:
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content.strip()
                    
                    if "Evaluator Feedback:" in content or "my job is" in content:
                        continue
                    
                    if '{' in content and '"action"' in content:
                        try:
                            start = content.find('{')
                            end = content.rfind('}') + 1
                            json_str = content[start:end]
                            
                            import json
                            parsed = json.loads(json_str)
                            
                            if all(key in parsed for key in ["reason", "content", "action"]):
                                return parsed
                        except json.JSONDecodeError:
                            continue
        
        return {
            "reason": "Failed to parse email response", 
            "content": "Could not extract decision", 
            "action": "ignore"
        }
        
    except Exception as e:
        print(f"[Error] Email decision extraction failed: {e}")
        return {"reason": f"Extraction error: {e}", "content": "", "action": "ignore"}

# --------- Gmail Integration ---------
def get_unread_email():
    """Fetch unread emails from Gmail API"""
    try:
        print("[Debug] Fetching Gmail messages...")
        service = get_gmail_service()
        if not service:
            print("[Error] Gmail service not available")
            return []

        results = service.users().messages().list(
            userId="me",
            labelIds=["INBOX"],
            q="is:unread"
        ).execute()

        messages = results.get("messages", [])
        print(f"[Debug] Found {len(messages)} unread emails")
        
        if not messages:
            return []

        emails = []
        for msg in messages[:5]:  # Limit to 5 most recent
            try:
                msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
                headers = msg_data.get("payload", {}).get("headers", [])
                subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No subject")
                sender = next((h["value"] for h in headers if h["name"] == "From"), "Unknown")

                emails.append({
                    "id": msg["id"],
                    "subject": subject,
                    "from": sender,
                    "snippet": msg_data.get("snippet", "")
                })
            except Exception as e:
                print(f"[Error] Failed to fetch email {msg['id']}: {e}")
                continue

        return emails

    except Exception as e:
        if "Gmail API has not been used" in str(e):
            print("[Error] Gmail API not enabled. Enable at: https://console.developers.google.com/apis/api/gmail.googleapis.com/overview")
        elif "403" in str(e):
            print("[Error] Gmail API access denied. Check credentials and permissions.")
        else:
            print(f"[Error] Gmail fetch failed: {e}")
        return []


# --------- Extract Agent Decision ---------
def extract_agent_decision(raw_response):
    """Extract the actual agent decision from LangGraph response"""
    try:
        print(f"[Debug] Raw response type: {type(raw_response)}")

        # Handle list of messages (LangGraph often returns [assistant, feedback])
        if isinstance(raw_response, (list, tuple)) and raw_response:
            # Find the first AI message that is NOT evaluator feedback
            for msg in raw_response:
                if hasattr(msg, "content"):
                    content = msg.content.strip()
                    
                    # Skip evaluator feedback explicitly
                    if "Evaluator Feedback:" in content:
                        print(f"[Debug] Skipping evaluator feedback")
                        continue
                    
                    # Try to parse as JSON
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # fallback plain text
                        return {"action": "notify_slack", "reason": "Plain text response", "content": content}
            
            # If we get here, all messages were evaluator feedback
            return {"action": "ignore", "reason": "Only evaluator feedback found", "content": ""}

        # Handle dict directly
        if isinstance(raw_response, dict):
            return normalize_decision(raw_response)

        # Handle string
        if isinstance(raw_response, str):
            return normalize_decision(raw_response)

        return {"action": "ignore", "reason": "Unknown format", "content": str(raw_response)}

    except Exception as e:
        print(f"[Error] extract_agent_decision failed: {e}")
        return {"action": "ignore", "reason": f"Extraction failed: {e}", "content": ""}



# --------- Normalize Decision ---------
def normalize_decision(decision):
    """Normalize agent output into a dict safely."""
    if isinstance(decision, dict):
        return {
            "reason": decision.get("reason", "No reason provided"),
            "content": decision.get("content", ""),
            "action": decision.get("action", "notify_slack")
        }

    if isinstance(decision, str):
        try:
            parsed = json.loads(decision)
            return {
                "reason": parsed.get("reason", "No reason provided"),
                "content": parsed.get("content", decision),
                "action": parsed.get("action", "notify_slack")            
            }
        except json.JSONDecodeError:
            return {
                "reason": "Invalid JSON from agent",
                "content": decision,
                "action": "notify_slack"
            }

    if isinstance(decision, (list, tuple)) and decision:
        return normalize_decision(decision[-1])

    return {"reason": "Unsupported type", "content": str(decision), "action": "ignore"}


# --------- Execute Actions ---------
async def execute_action(decision, context_data=None):
    """Execute actions based on decision"""
    try:
        action = decision.get("action")
        content = decision.get("content", "")
        
        print(f"[Debug] Executing action: {action}")
        
        if action == "ignore":
            print("[Debug] actions ignored")
            return

        if action == "reply_email" and context_data:
            try:
                service = get_gmail_service()
                if not service:
                    print("[Error] Gmail service not available for reply")
                    return
                meeting_response = await book_meeting_if_detected(context_data if isinstance(context_data , list) else [context_data])
                if meeting_response:
                    content  = f"{content}\n\n{meeting_response}"
                    print(f"[debug] Added meeting bookin to replay email")


                from email.mime.text import MIMEText
                import base64

                # Get the list of emails
                emails_list = context_data if isinstance(context_data, list) else [context_data]
                
                # Filter out newsletters and automated emails first
                filtered_emails = []
                for email in emails_list:
                    email_from = email.get("from", "").lower()
                    email_subject = email.get("subject", "").lower()
                    
                    # Skip newsletters, automated emails, and no-reply addresses
                    skip_patterns = [
                        "newsletter", "no-reply", "noreply", "donotreply", 
                        "@update.", "mlive.com", "automated", "system@",
                        "admin@", "support@", "alerts@", "notifications@"
                    ]
                    
                    if not any(pattern in email_from for pattern in skip_patterns) and \
                       not any(pattern in email_subject for pattern in ["unsubscribe", "newsletter", "top stories", "daily digest"]):
                        filtered_emails.append(email)
                
                if not filtered_emails:
                    print("[Error] No suitable emails found for reply after filtering")
                    return
                
                # Use the first important email (the LLM should have chosen the most important one)
                target_email = filtered_emails[0]
                
                print(f"[Debug] Replying to: {target_email['from']} - Subject: {target_email['subject']}")
                
                # Create the reply
                msg = MIMEText(content)
                msg["to"] = target_email["from"]
                msg["subject"] = f"{target_email['subject']}"
                msg["from"] = os.getenv("EMAIL_USER")
                
                # Add In-Reply-To and References headers for proper threading
                if "id" in target_email:
                    msg["In-Reply-To"] = f"<{target_email['id']}>"
                    msg["References"] = f"<{target_email['id']}>"

                raw = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
                message = {"raw": raw}

                # Send the email
                result = service.users().messages().send(userId="me", body=message).execute()
                message_id = result.get('id', 'unknown')
                
                print(f"[SUCCESS] Email sent successfully!")
                print(f"  To: {target_email['from']}")
                print(f"  Subject: Re: {target_email['subject']}")
                print(f"  Message ID: {message_id}")
                print(f"  Reply content: {content[:100]}...")
                
                await slack_notification_async(f"📨 Successfully replied to {target_email['from'][:30]}: {content[:50]}...")

            except Exception as e:
                print(f"[Error] Failed to send email reply: {e}")
                import traceback
                print(f"[Debug] Full error traceback: {traceback.format_exc()}")
                await slack_notification_async(f"❌ Email reply failed: {str(e)}")

        # 2. Reschedule Calendar Event
        elif action == "reschedule_event":
            try:
                from ai_sidekick.calendar_integration import reschedule_event
                reschedule_event(content)
                print(f"📅 Rescheduled event: {content}")
                await slack_notification_async(f"📅 Event Rescheduled: {content}")
            except Exception as e:
                print(f"[Error] Failed to reschedule event: {e}")
                await slack_notification_async(f"❌ Failed to reschedule event: {str(e)}")

        # 3. Notify via Slack (only for meaningful content)
        elif action == "notify_slack" and content.strip():
            # Skip evaluator feedback and generic responses
            if not any(skip in content for skip in ["Evaluator Feedback:", "my job is", "I'm designed to"]):
                await slack_notification_async(content)
                print("📢 Slack notification sent")
            else:
                print("[Debug] Skipped generic/feedback notification")

        else:
            print(f"[Debug] No action taken for: {action}")
            
    except Exception as e:
        print(f"[Error] execute_action failed: {e}")


# --------- Memory Functions ---------
def save_memory(data, file="memory.json"):
    try:
        try:
            with open(file, 'r') as f:
                memory = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            memory = []
        
        memory.append(data)
        
        # Keep only last 50 entries
        if len(memory) > 50:
            memory = memory[-50:]
        
        with open(file, "w") as f:
            json.dump(memory, f, indent=2)
            
    except Exception as e:
        print(f"[Error] Failed to save memory: {e}")


def load_memory(file="memory.json"):
    try:
        with open(file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


# --------- Calendar Integration ---------
async def check_calendar():
    try:
        events = list_events()
        if not events:
            return
            
        now = datetime.now(timezone.utc)
        for event in events:
            start = parser.parse(event["start"])
            delta = start - now
            
            # Notify 1 hour before
            if timedelta(minutes=59) <= delta <= timedelta(hours=1, minutes=1):
                message = f"📅 Reminder: '{event['summary']}' starts in 1 hour at {event['start']}"
                await slack_notification_async(message)
                print(f"📅 Calendar reminder sent for: {event['summary']}")
                
    except Exception as e:
        print(f"[Error] Calendar check failed: {e}")


# --------- News API Integration ---------
def get_ai_news():
    """Fetch AI news from NewsAPI"""
    try:
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            print("[Warning] NEWS_API_KEY not found")
            return []

        url = f"https://newsapi.org/v2/everything?q=artificial intelligence OR machine learning OR AI&sortBy=publishedAt&apiKey={api_key}&pageSize=5"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get("status") == "ok":
            articles = data.get("articles", [])
            print(f"[Debug] Fetched {len(articles)} news articles")
            return articles
        else:
            print(f"[Error] News API error: {data.get('message', 'Unknown error')}")
            return []
            
    except Exception as e:
        print(f"[Error] News fetch failed: {e}")
        return []


def categorize_articles(title: str) -> str:
    """Return an emoji based on title"""
    title_lower = title.lower()

    if any(company in title_lower for company in ["meta", "google", "microsoft", "apple", "amazon"]):
        return "🏢"
    elif any(ai_term in title_lower for ai_term in ["ai", "artificial intelligence", "machine learning", "neural", "chatgpt", "openai"]):
        return "🤖"
    elif any(market_term in title_lower for market_term in ["stock", "market", "invest", "fund", "ipo"]):
        return "📈"
    elif any(global_term in title_lower for global_term in ["global", "world", "international"]):
        return "🌎"
    return "📰"


# --------- Main Agent Loop ---------
async def watcher_loop():
    try:
        print("🚀 Starting AI Watcher Agent...")
        agent = Sidekick()
        await agent.setup()

        last_news_check = datetime.now(timezone.utc) - timedelta(hours=1)  # Check news immediately
        last_email_check = datetime.now(timezone.utc)
        processed_emails = set()  # Track processed emails
        
        history = []

        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                # --- 1. Email Processing (every 2 minutes) ---
                if (current_time - last_email_check).total_seconds() >= 120:
                    print("\n--- Checking Emails ---")
                    emails = get_unread_email()
                    
                    # Filter out already processed emails
                    new_emails = [email for email in emails if email['id'] not in processed_emails]
                    
                    if new_emails:
                        print(f"[Debug] Processing {len(new_emails)} new emails")
                        
                        # Show all emails for debugging
                        for i, email in enumerate(new_emails):
                            print(f"[Debug] Email {i+1}: From={email.get('from')}, Subject={email.get('subject')}")
                        
                        # Filter out newsletters and automated emails
                        important_emails = []
                        for email in new_emails:
                            email_from = email.get("from", "").lower()
                            email_subject = email.get("subject", "").lower()
                            
                            # Skip patterns for automated/newsletter emails
                            skip_from_patterns = [
                                "newsletter", "no-reply", "noreply", "donotreply",
                                "@update.", "mlive.com", "automated", "system@",
                                "admin@", "alerts@", "notifications@", "marketing@"
                            ]
                            
                            skip_subject_patterns = [
                                "unsubscribe", "newsletter", "top stories", "daily digest",
                                "weekly update", "promotional", "marketing"
                            ]
                            
                            # Check if this email should be skipped
                            should_skip = (
                                any(pattern in email_from for pattern in skip_from_patterns) or
                                any(pattern in email_subject for pattern in skip_subject_patterns)
                            )
                            
                            if not should_skip:
                                important_emails.append(email)
                                print(f"[Debug] Kept important email from: {email_from}")
                            else:
                                print(f"[Debug] Filtered out automated email from: {email_from}")
                        
                        if not important_emails:
                            print("[Debug] No important emails found after filtering")
                            # Mark all as processed since they're just newsletters
                            for email in new_emails:
                                processed_emails.add(email['id'])
                        else:
                            print(f"[Debug] Found {len(important_emails)} important emails to analyze")
                            
                            # Create summary of important emails only
                            email_summary = "\n".join([
                                f"From: {email['from']}\nSubject: {email['subject']}\nSnippet: {email['snippet'][:100]}...\n"
                                for email in important_emails[:3]  # Limit to top 3 important emails
                            ])
                            
                            try:
                                prompt = f"""
You are an email analyzer. Analyze these {len(important_emails)} important emails and choose ONE action.

IMPORTANT EMAILS:
{email_summary}

INSTRUCTIONS:
1. Pick the MOST IMPORTANT email that requires human attention
2. Decide the best action for that email
3. Respond with ONLY valid JSON in this EXACT format:

{{"reason": "Brief explanation of why this email is important", "content": "Your professional response message", "action": "reply_email|notify_slack|ignore"}}

ACTIONS:
- reply_email: If the email requires a professional response (provide the full reply text in "content")
- notify_slack: If you just need to alert the user about the email (provide alert message in "content") 
- ignore: If none of the emails require immediate action

RESPOND WITH ONLY THE JSON OBJECT, NO OTHER TEXT.
"""

                                from langchain_core.messages import SystemMessage, HumanMessage
                                
                                simple_response = await agent.worker_llm_with_tools.ainvoke([
                                    SystemMessage(content="You are an email analyzer. Respond with valid JSON only."),
                                    HumanMessage(content=prompt)
                                ])
                                
                                print(f"[Debug] LLM email response: {simple_response.content[:200]}...")
                                
                                if hasattr(simple_response, 'content'):
                                    content = simple_response.content.strip()
                                    if '{' in content and '"action"' in content:
                                        try:
                                            start = content.find('{')
                                            end = content.rfind('}') + 1
                                            json_str = content[start:end]
                                            
                                            decision = json.loads(json_str)
                                            print(f"[Debug] Email decision: {decision}")
                                            
                                            if decision.get("action") == 'reply_email' and decision.get("content"):
                                                await execute_action(decision, important_emails)
                                            elif decision.get("action") == 'notify_slack' and decision.get("content"):
                                                await slack_notification_async(decision['content'])
                                                print("Slack notification sent")                                        
                                            # Mark ALL original emails as processed (including filtered ones)
                                            for email in new_emails:
                                                processed_emails.add(email['id'])
                                                
                                        except json.JSONDecodeError as e:
                                            print(f"[Error] JSON parsing failed: {e}")
                                    else:
                                        print("[Debug] No valid JSON found in email response")
                                else:
                                    print("[Debug] No content in email response")
                                    
                            except Exception as e:
                                print(f"[Error] Email processing failed: {e}")
                    else:
                        print("[Debug] No new emails to process")

                # --- 2. Calendar Check ---
                await check_calendar()

                # --- 3. News Check (every 30 minutes) ---
                if (current_time - last_news_check).total_seconds() >= 1800:  # 30 minutes
                    print("\n--- Checking News ---")
                    articles = get_ai_news()
                    
                    if articles:
                        article_summary = "\n".join([
                            f"{categorize_articles(article['title'])} {article['title']}"
                            for article in articles[:5]
                        ])

                        try:
                            prompt = f"""
You are a professional news anchor delivering a biref updates. Create a conversational, authoritative news summary from these articles:

{article_summary}

Write as if you're speaking directly to viewers during a live broadcast. Use a confident, informed tone like major news networks. Focus on the 2-3 most significant stories that would impact your audience.

Guidelines:
- Sound like a seasoned news professional 
- Use clear, direct language
- Include relevant context when needed
- Keep it conversational but authoritative
- Maximum 180 characters for Slack
- Add appropriate emojis naturally

Example tone: "Breaking developments in AI technology today as major companies announce..." or "In tech news this hour..."

Deliver only the news summary - no formatting, no explanations.

"""
                            from langchain_core.messages import HumanMessage, SystemMessage
                            response = await agent.worker_llm_with_tools.ainvoke([
                                SystemMessage(content= "Create a professional news digest for slack. Respond only the message content"),
                                HumanMessage(content=prompt)
                            ])

                            content = response.content.strip()
                            print(f"[Debug] News llm response: {content}")

                            if content and len(content) <300 and not any(skip in content.lower() for skip in ["requirements", "json format", "evaluator feedback", "resppond only"]):
                                  try:
                                      await slack_notification_async(content)
                                      print(f"Slack Notification sent: {content[:50]}")
                                  except Exception as e:
                                      print(f"Failed to send salck notification{e}")
                            else:
                                 print(f"[Debug] Skipped Evaluator Feedback")

                            save_memory({
                                    "timestamp": current_time.isoformat(),
                                    "type": "news",
                                    "articles_count": len(articles),
                                    "content_preview": content[:100].replace('"', "'").replace("{", '[').replace("}",']')
                                })

                            history.append(f"news_processed: {len(articles)}")
                            
                        except Exception as e:
                            print(f"[Error] News processing failed: {e}")
                            
                    last_news_check = current_time

                # --- 4. Cleanup and Sleep ---
                # Clean up history to prevent memory issues
                if len(history) > 10:
                    history = history[-5:]
                
                # Clean up processed emails set
                if len(processed_emails) > 100:
                    processed_emails = set(list(processed_emails)[-50:])
                
                print(f"💤 Sleeping... Next check in 2 minutes")
                await asyncio.sleep(120)  # 2 minutes
                
            except Exception as e:
                print(f"[Error] Loop iteration failed: {e}")
                await asyncio.sleep(60)  # 1 minute on error
                
    except Exception as e:
        print(f"[Fatal Error] Watcher loop crashed: {e}")
        # Try to restart after 5 minutes
        await asyncio.sleep(300)
        await watcher_loop()


if __name__ == "__main__":
    print("🤖 AI Personal Assistant Watcher Starting...")
    try:
        asyncio.run(watcher_loop())
    except KeyboardInterrupt:
        print("\n👋 Watcher stopped by user")
    except Exception as e:
        print(f"[Fatal] Failed to start: {e}")