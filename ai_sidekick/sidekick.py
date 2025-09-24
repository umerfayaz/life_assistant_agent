from ast import keyword
from re import escape
from .agent_config import AGENT_NAME, USER_NAME, USER_PASSWORD, personal_commands, agent_knowledge
from typing import Annotated, List, Any, Optional, Dict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from ai_sidekick.sidekick_tools import playwright_tools, other_tools
from ai_sidekick.sidekick_tools import slack_notification_tool
import uuid
import os
import asyncio
from datetime import datetime
import json

load_dotenv(override=True)

class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )


class Sidekick:
    def __init__(self):
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.tools = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())
        self.memory = MemorySaver()
        self.browser = None
        self.playwright = None

    async def setup(self):
        """Setup the agent with tools and LLMs"""
        try:
            self.tools, self.browser, self.playwright = await playwright_tools()
            additional_tools = await other_tools()
            additional_tools.append(slack_notification_tool)
            self.tools += additional_tools
            print(f"✓ Loaded {len(self.tools)} tools successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load tools: {e}")
            self.tools = []

        try:
            worker_llm = ChatGroq(
                model="openai/gpt-oss-120b",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.7,
                max_tokens=4096,
                # stop_sequences=["<function=", "</function>", "function="]
            )

            if self.tools:
                self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)
            else:
                self.worker_llm_with_tools = worker_llm

            evaluator_llm = ChatGroq(
                model="openai/gpt-oss-120b",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.7,
            )
            self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
            print("✓ LLMs configured successfully")

            await self.build_graph()
            print("✓ Graph built successfully")

        except Exception as e:
            print(f"✗ Error in setup: {e}")
            raise

    PLANNER_SYSTEM_PROMPT = """
You are the Planner agent in a LangGraph workflow. 
Your job is to decide the next step based on the user’s request or the latest task result.

Available next steps (nodes, not tools!):
- worker → reasoning / execution
- tools → when you need an actual tool (search, email, flight booking, Slack notification, etc.)
- evaluator → review output and decide next actions
- slack → send a user-facing Slack notification
- END → finish workflow


Special Slack instructions:
- If the output is news summaries, alerts, confirmations, or user-facing updates, 
  set "next": "slack".
- Always keep Slack messages short (under 200 characters) and engaging. Use emojis where relevant.
- For backend logic or intermediate steps, route to worker or evaluator instead.

Your response must be a JSON object with this format:
{
  "next": "worker" | "tools" | "evaluator" | "slack" | "END",
  "content": "...summary or message for the next node..."
}
"""

    async def planner(self, state: State) -> Dict[str, Any]:
        """Planner that decides what the agent should do next"""
        last_message = state["messages"][-1].content if state["messages"] else ""
    
        # Create a more specific prompt for the planner
        planner_prompt = f"""
        You are a Planner agent. Analyze the current state and decide the next action.
        
        Current state: {state}
        Last message: {last_message}
    
    Available actions:
    - worker: For reasoning, processing, or general responses
    - tools: When specific tools need to be called (search, email, calendar, etc.)
    - evaluator: To review and evaluate responses
    - slack: For sending user-facing notifications, news digests, alerts, or confirmations
    - END: To finish the workflow
    
    Respond with ONLY a JSON object in this exact format:
    {{"next": "worker|tools|evaluator|slack|END", "content": "brief explanation"}}
        """
        
        decision = await self.worker_llm_with_tools.ainvoke([
            SystemMessage(content=planner_prompt),
            HumanMessage(content=f"Decide next action based on: {last_message}")
        ])
        
        try:
            # Try to parse JSON from the response
            import json
            response_content = decision.content.strip()
            
            # Clean up the response if it has extra text
            if '{' in response_content and '}' in response_content:
                start = response_content.find('{')
                end = response_content.rfind('}') + 1
                json_str = response_content[start:end]
                parsed = json.loads(json_str)
                
                next_step = parsed.get("next", "worker").lower()
                if next_step not in ["worker", "tools", "evaluator", "slack", "end"]:
                    next_step = "worker"
                    
                return {"next": next_step, "content": parsed.get("content", "")}
            else:
                # Fallback if JSON parsing fails
                return {"next": "worker", "content": "Routing to worker"}
                
        except Exception as e:
            print(f"Planner parsing error: {e}")
            return {"next": "worker", "content": "Fallback to worker"}

    def worker(self, state: State) -> Dict[str, Any]:
        """
        Main worker function that processes user requests
        """
        messages = state["messages"]
        # Extract last human message
        user_message = ""

        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        user_input_lower = user_message.lower()
        
        if "hey" in user_input_lower:
            return {"messages":[AIMessage(content= f"hey there how are you today ? 😊")]}

        if "your name" in user_input_lower or "who are you" in user_input_lower:
          return {"messages": [AIMessage(content=f"My name is {AGENT_NAME}. Created by Umer to help with daily tasks.")]}

        if "my name" in user_input_lower:
           return {"messages": [AIMessage(content=f"Your name is {USER_NAME}.")] }

        # 1️⃣ Knowledge questions
        for key, value in agent_knowledge.items():
            if key in user_input_lower:
                return {"messages": [AIMessage(content=f"my {key} is {value}.")]}

        # 2️⃣ Personal info request (password protected)
        if "my info" in user_input_lower or "personal" in user_input_lower:
            pwd = state.get("password")
            if not pwd:
                return {"messages": [AIMessage(content="Please provide your password to access personal info")]}

            if pwd == USER_PASSWORD:
                return {"messages":[AIMessage(content = "Welcome Umer here's your Data: [memory data]")]}
            else:
                return {"messages":[AIMessage(content = "Incorrect Password. cannot share personal info.")]}

        # 3️⃣ Personal commands
        for cmd, desc in personal_commands.items():
            if cmd in user_input_lower:
                return {"messages": [AIMessage(content=f"Executing Method: {desc}")]}

        if "show_my_meetings" in user_input_lower or "my appointments" in user_input_lower:
            get_tools = next((t for t in self.tools if t.name == "get_meetings"),None)
            if get_tools:
                meeting = self.get_tools.invoke({})
                return {"messages":[AIMessage(content= f"Here are the upcoming meeting schedule:\n{meeting}")]}
            else:
                return {"messages": [AIMessage(content= f"Get meeting tools are not available")]}

        if any(keyword in user_input_lower for keyword in ["reschedule", "change", "update" ]) and "meeting" in user_input_lower:
            details = self.extract_meeting_details(user_message)
            if details:
                reschedule_tools = next((t for t in self.tools if t.name == "reschedule_meeting"))
                if reschedule_tools:
                    result = reschedule_tools.invoke(details)
                    return {"messages":[AIMessage(content = f"Your meeting schedule had been updates successfully:\n{result}")]}
                else:
                    return {"messages": [AIMessage(content= f"Rescedule tool is not available")]}
       
        # Check for SENDING email triggers (actually send the email)

        send_email_triggers = [
            "send this email", "send the email", "email this to me", 
            "send me this email", "send this via email", "email me this",
            "send it to my email", "email me the details", "email me"
        ]
        
        # Check for WRITING email triggers (just generate content, don't send)
        write_email_triggers = [
            "write email", "compose email", "draft email", 
            "create email", "generate email", "make email"
        ]
        
        if any(trigger in user_input_lower for trigger in send_email_triggers):
            # User wants to SEND an email - look for previous content to send
            try:
                to_email = os.getenv("EMAIL_USER")
                
                # Look for previous assistant message with content to email
                previous_content = ""
                for msg in reversed(messages[:-1]):  # Skip current message
                    if isinstance(msg, AIMessage) and msg.content:
                        # Skip evaluator feedback messages
                        if not msg.content.startswith("Evaluator Feedback:"):
                            previous_content = msg.content
                            break
                
                if previous_content:
                    # Send the previous content as email
                    if "flight" in user_input_lower or "booking" in user_input_lower:
                        email_subject = "Flight Information"
                    else:
                        email_subject = "Information from Mindara Assistant"
                    
                    # Use the previous content as email body
                    email_body = f"Hello,\n\nHere's the information you requested:\n\n{previous_content}\n\nBest regards,\n{AGENT_NAME}"
                    
                    result = self.send_email_notification(
                        to_email=to_email,
                        subject=email_subject,
                        body=email_body
                    )
                    
                    return {"messages": [AIMessage(content=f"I've sent you an email check it out! {result}")]}
                else:
                    return {"messages": [AIMessage(content="I don't have any previous content to email you. Could you please specify what you'd like me to send?")]}
                    
            except Exception as e:
                return {"messages": [AIMessage(content=f"❌ Failed to send email: {str(e)}")]}
        
        elif any(trigger in user_input_lower for trigger in write_email_triggers):
            # User wants to WRITE/DRAFT an email - generate content but don't send
            try:
                if "flight" in user_input_lower or "booking" in user_input_lower:
                    email_subject = "Flight Information Request"
                    email_prompt = f"""
                    Based on this user request: "{user_message}"
                    
                    Generate a professional email about flight information. Include:
                    - Greeting
                    - Summary of the flight request/booking details
                    - Any relevant flight information
                    - Professional closing
                    
                    Keep it concise and professional.
                    """
                else:
                    email_subject = "Information Request"
                    email_prompt = f"""
                    Based on this user request: "{user_message}"
                    
                    Generate a professional email that addresses their request. Include:
                    - Appropriate greeting
                    - Main content addressing their request
                    - Professional closing
                    
                    Make it helpful and professional.
                    """
                
                email_content_messages = [
                    SystemMessage(content="You are an email writer. Generate professional email content based on the user's request."),
                    HumanMessage(content=email_prompt)
                ]
                
                email_response = self.worker_llm_with_tools.invoke(email_content_messages)
                email_body = email_response.content if hasattr(email_response, 'content') else str(email_response)
                
                return {"messages": [AIMessage(content=f"📧 Here's your email draft:\n\nSubject: {email_subject}\n\n{email_body}\n\n💡 Say 'send this email' or 'email me this' if you want me to send it to you!")]}
                
            except Exception as e:
                return {"messages": [AIMessage(content=f"❌ Failed to generate email: {str(e)}")]}

        try:
            tool_names = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"
            
            system_message = f"""
You are {AGENT_NAME}, a personal AI assistant for {USER_NAME}.
You respond naturally and conversationally like a human.

CORE BEHAVIOR:
- If a user asks for something that requires a tool ({tool_names}), you MUST:
  1. Produce a natural, conversational reply for the user (e.g., "Okay, I'll schedule that for you").
  2. ALSO produce an invisible structured tool call in the background.

RULES FOR CALENDAR:
- If the user asks to book, schedule, create, or manage an event/meeting, you MUST use the calendar tool (`calendar_booking`) instead of just replying in text.
- Always confirm missing details (title, date, time, duration, time zone, participants) before booking.
- Once details are clear, ALWAYS trigger the `calendar_booking` tool.
- Do NOT simulate a booking in text only. If booking is possible, call the tool.
- Never show raw JSON, tool names, or system calls to the user.

GENERAL:
- Always provide BOTH:
  1. A friendly, natural response for the user.
  2. An invisible, structured tool call for the system.
"""




            if state.get("feedback_on_work"):
                system_message += f"""

Previous attempt feedback: {state['feedback_on_work']}
Please address this feedback in your response.
"""

            found_system_message = False
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    msg.content = system_message
                    found_system_message = True
                    break

            if not found_system_message:
                messages = [SystemMessage(content=system_message)] + messages

            response = self.worker_llm_with_tools.invoke(messages)
            print("DEBUG LLM Response:", getattr(response, "content", None))
            print("DEBUG Tool Calls:", getattr(response, "tool_calls", None))

            return {"messages": [response]}

        except Exception as e:
            print(f"Worker error: {e}")
            error_response = AIMessage(
                content="I encountered an issue processing your request. Could you please rephrase or provide more details?"
            )
            return {"messages": [error_response]}

    def extract_meeting_details(self, message):
        """Extract meeting details from user message"""
        try:
            import re
            from datetime import datetime, timedelta
            
            # Look for meeting title/summary
            summary_patterns = [
                r"meeting (?:about|with|for) (.+?)(?:\s+on|\s+at|\s+from|\s+tomorrow|\s+today|$)",
                r"book (.+?)(?:\s+meeting|\s+on|\s+at|\s+from|\s+tomorrow|\s+today|$)",
                r"schedule (.+?)(?:\s+meeting|\s+on|\s+at|\s+from|\s+tomorrow|\s+today|$)",
            ]
            
            summary = "Meeting"  # default
            for pattern in summary_patterns:
                match = re.search(pattern, message.lower())
                if match:
                    summary = match.group(1).strip().title()
                    break
            
            # Look for times - handle both ISO format and natural language
            time_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})'
            times = re.findall(time_pattern, message)
            
            start_time = None
            end_time = None
            
            if len(times) >= 2:
                start_time = times[0]
                end_time = times[1]
            elif len(times) == 1:
                start_time = times[0]
                # If only start time provided, assume 1 hour duration
                try:
                    start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = start_dt + timedelta(hours=1)
                    end_time = end_dt.strftime('%Y-%m-%dT%H:%M:%S')
                except:
                    pass
            else:
                # Try to parse natural language times
                if "tomorrow at" in message.lower():
                    time_match = re.search(r'tomorrow at (\d{1,2}):?(\d{2})?\s*(am|pm)?', message.lower())
                    if time_match:
                        hour = int(time_match.group(1))
                        minute = int(time_match.group(2)) if time_match.group(2) else 0
                        am_pm = time_match.group(3)
                        
                        if am_pm == 'pm' and hour != 12:
                            hour += 12
                        elif am_pm == 'am' and hour == 12:
                            hour = 0
                        
                        tomorrow = datetime.now() + timedelta(days=1)
                        start_dt = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        end_dt = start_dt + timedelta(hours=1)
                        
                        start_time = start_dt.strftime('%Y-%m-%dT%H:%M:%S')
                        end_time = end_dt.strftime('%Y-%m-%dT%H:%M:%S')
            
            if not start_time or not end_time:
                return None
            
            # Look for description
            description = ""
            desc_patterns = [
                r"about (.+?)(?:\s+tomorrow|\s+today|\s+at|\s+from|$)",
                r"discuss (.+?)(?:\s+tomorrow|\s+today|\s+at|\s+from|$)",
            ]
            
            for pattern in desc_patterns:
                match = re.search(pattern, message.lower())
                if match:
                    description = match.group(1).strip()
                    break
            
            return {
                "summary": summary,
                "description": description,
                "start_time": start_time,
                "end_time": end_time
            }
            
        except Exception as e:
            print(f"Error extracting meeting details: {e}")
            return None

    def worker_router(self, state: State) -> str:
        try:
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print(f"Routing to tools: {len(last_message.tool_calls)} tool calls detected")
                return "tools"

            if hasattr(last_message, "content") and last_message.content:
                content = str(last_message.content)
                if "<function=" in content or content.startswith("function="):
                    print("Malformed function call detected, routing to evaluator")

            return "evaluator"

        except Exception as e:
            print(f"Router error: {e}")
            return "evaluator"

    def send_email_notification(self, to_email: str, subject: str, body: str) -> str:
        if not self.tools:
            return "Tools not loaded, cannot send email"

        tool = next((t for t in self.tools if t.name == "send_email"), None) 
        if not tool:
            return "Email tool not found"

        result = tool.invoke({
            "subject": subject,
            "body": body,
            "to_email": to_email
        })

        return result


    def evaluator(self, state: State) -> Dict[str, Any]:
        try:
            last_message = state["messages"][-1]
            last_response = getattr(last_message, "content", str(last_message))

            system_message = """You are an evaluator that checks if the assistant met the success criteria.

EVALUATION RULES:
- If the response contains raw function calls or malformed syntax, mark as failed
- If the response is helpful and addresses the user's request, mark as successful
- If more clarification is needed, mark user_input_needed as True
"""

            user_message = f"""
Conversation history:
{self.format_conversation(state['messages'])}

Success criteria:
{state['success_criteria']}

Assistant's latest response:
{last_response}

Please evaluate if the success criteria is met and provide feedback.
"""

            evaluator_messages = [
                SystemMessage(content=system_message),
                HumanMessage(content=user_message),
            ]

            eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)

            return {
                "messages": [AIMessage(content=f"Evaluator Feedback: {eval_result.feedback}")],
                "feedback_on_work": eval_result.feedback,
                "success_criteria_met": eval_result.success_criteria_met,
                "user_input_needed": eval_result.user_input_needed,
            }

        except Exception as e:
            print(f"Evaluator error: {e}")
            return {
                "messages": [AIMessage(content="Evaluation error occurred.")],
                "feedback_on_work": "Technical error during evaluation",
                "success_criteria_met": False,
                "user_input_needed": True,
            }

    def route_based_on_evaluation(self, State: State) ->str:

        try:
            if State.get("success_criteria_met"):

                assistant_content = None
                for msg in reversed(State.get("messages", [])):
                    if hasattr(msg, 'content') and msg.content:
                        content = msg.content

                        if not content.startswith("Evaluator Feedback"):
                            assistant_content = content
                            break


                        if assistant_content:
                            content_lower = assistant_content.lower()
                            if any(keyword in content_lower for keyword in [
                                "news", "completed", "alert", "scheduled",
                                "meeting", "digesr", "booked", "confirmed",                
                            ]):

                             State["slack_content"] = assistant_content

            if State.get["success_criteria_met"] and State.get["user_input_needed"]:
                return "END"
            return "worker"
        except Exception as e:
            print(f"Evaluator routing error: {e}")
            return "END"   

    def format_conversation(self, messages: List[Any]) -> str:
        conversation = ""
        for msg in messages:
            try:
                if isinstance(msg, HumanMessage):
                    conversation += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    content = getattr(msg, "content", "[Tool use]") or "[Tool use]"
                    conversation += f"Assistant: {content}\n"
            except Exception as e:
                print(f"Error formatting message: {e}")
                conversation += "Assistant: [Error formatting message]\n"
        return conversation


    async def slack_node(self, State: State) -> Dict[str, Any]:
     """Handle Sending Slack Notifications"""

     try:
        message_to_send = None

        if "slack_content" in State:
            message_to_send = ["slack_content"]


        elif State.get("messsages"):
            for msg in reversed(State["messages"]):
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content

                    if (not content.startswith("Evaluator Feedback:")and 
                    not content.startswith("slack notification sent:")and 
                    not content.startswith("System:")and
                    " Ai digest:" in content or
                    "digest" in content.lower()or
                    "news" in content.lower()):

                     message_to_send = content
                     break

        if not message_to_send and State.get("messages"):
            for msg in reversed(State["messages"]):
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content
                    if (not content.startswith("Evaluator Feedback")and
                    not content.startswith("slack notification sent")and
                    len(content.stripe())> 0):
                     message_to_send = content
                    break
        
        if not message_to_send and content in State:
            message_to_send = State["content"]

        if not message_to_send:
            message_to_send = "No Messsage Content found"

        print(f"[Debug] Slack Node is sending {message_to_send[:100]}...")

        result = slack_notification_tool(message_to_send)

        return {
            "messages": [AIMessage(content = f"Slack notification sent Successfully{result}")],
            "last_output": result
        }

     except Exception as e:
        print(f"Failed to send slack notification{e}")
        
        return {
            "messages": [AIMessage(content = f" Failed to send slack notifications{str(e)}")],
            "last_message": f"Error{str(e)}"
        }


    async def build_graph(self):
        """
        Build the LangGraph workflow graph
        """
        try:
            graph_builder = StateGraph(State)

            # Add nodes
            graph_builder.add_node("planner", self.planner)
            graph_builder.add_node("worker", self.worker)
            graph_builder.add_node("evaluator", self.evaluator)
            graph_builder.add_node("slack", self.slack_node)

            # Add tools node if tools are available
            if self.tools:
                graph_builder.add_node("tools", ToolNode(tools=self.tools))
                
                # Add conditional edges from worker
                graph_builder.add_conditional_edges(
                    "worker", 
                    self.worker_router, 
                    {"tools": "tools", "evaluator": "evaluator"}
                )
                
                # Add edge from tools back to worker
                graph_builder.add_edge("tools", "planner")
            else:
                # If no tools, go directly to evaluator
                graph_builder.add_edge("worker", "evaluator")

            # planner routes to next node
            graph_builder.add_conditional_edges(
                "planner",
                lambda State: State.get("next", "worker"),
                {"worker": "worker", "tools": "tools", "evaluator": "evaluator", "slack": "slack", "end": END}
            )

            # Add conditional edges from evaluator
            graph_builder.add_conditional_edges(
                "evaluator", 
                self.route_based_on_evaluation, 
                {"worker": "worker","slack": "slack", "END": END}
            )

            graph_builder.add_edge("slack", END)
            
            # Set start edge
            graph_builder.add_edge(START, "planner")

            # Compile the graph with memory
            self.graph = graph_builder.compile(checkpointer=self.memory)

        except Exception as e:
            print(f"✗ Graph building error: {e}")
            raise


    async def run_superstep(
        self, 
        message: str, 
        success_criteria: str, 
        history: List[Dict], 
        _allowed_tools: Optional[List[str]] = None
    ):
        try:
            state = {
                "messages": [HumanMessage(content=message)],
                "success_criteria": success_criteria or "The answer should be clear and accurate",
                "feedback_on_work": None,
                "success_criteria_met": False,
                "user_input_needed": False,
            }

            config = {"configurable": {"thread_id": self.sidekick_id}}

            result = await self.graph.ainvoke(state, config=config)

            messages = result.get("messages", [])
            user_msg = {"role": "user", "content": message}

            print(f"[Debug] Messages length: {len(messages)}")
            print(f"[Debug] Messages types: {[type(msg) for msg in messages]}")
            
            if len(messages) >= 2:
                # Safely extract content from message objects
                assistant_msg_obj = messages[-2]
                feedback_msg_obj = messages[-1]
                
                print(f"[Debug] Assistant message type: {type(assistant_msg_obj)}")
                print(f"[Debug] Feedback message type: {type(feedback_msg_obj)}")
                
                # Safe content extraction
                if hasattr(assistant_msg_obj, 'content'):
                    assistant_content = assistant_msg_obj.content
                elif isinstance(assistant_msg_obj, dict):
                    assistant_content = assistant_msg_obj.get('content', 'No response')
                elif isinstance(assistant_msg_obj, str):
                    assistant_content = assistant_msg_obj
                else:
                    assistant_content = str(assistant_msg_obj)
                
                if hasattr(feedback_msg_obj, 'content'):
                    feedback_content = feedback_msg_obj.content
                elif isinstance(feedback_msg_obj, dict):
                    feedback_content = feedback_msg_obj.get('content', 'No feedback')
                elif isinstance(feedback_msg_obj, str):
                    feedback_content = feedback_msg_obj
                else:
                    feedback_content = str(feedback_msg_obj)

                assistant_msg = {"role": "assistant", "content": assistant_content}
                feedback_msg = {"role": "system", "content": feedback_content}

                print(f"[Debug] Extracted assistant content: {assistant_content[:100]}...")
                print(f"[Debug] Extracted feedback content: {feedback_content[:100]}...")

                return history + [user_msg, assistant_msg, feedback_msg]
                
            elif len(messages) == 1:
                # Handle single message case
                single_msg_obj = messages[0]
                
                if hasattr(single_msg_obj, 'content'):
                    content = single_msg_obj.content
                elif isinstance(single_msg_obj, dict):
                    content = single_msg_obj.get('content', 'No response')
                elif isinstance(single_msg_obj, str):
                    content = single_msg_obj
                else:
                    content = str(single_msg_obj)
                    
                assistant_msg = {"role": "assistant", "content": content}
                return history + [user_msg, assistant_msg]
                
            else:
                print("[Debug] No messages in result, using fallback")
                error_msg = {"role": "assistant", "content": "I encountered an issue processing your request."}
                return history + [user_msg, error_msg]

        except Exception as e:
            print(f"Run superstep error: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            
            user_msg = {"role": "user", "content": message}
            error_msg = {"role": "assistant", "content": f"Error: {str(e)}"}
            return history + [user_msg, error_msg]
    

    def cleanup(self):
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                try:
                    asyncio.run(self.browser.close())
                    if self.playwright:
                        asyncio.run(self.playwright.stop())
                except Exception as e:
                    print(f"Cleanup error: {e}")
            except Exception as e:
                print(f"Cleanup error: {e}")




              



 





    








