from re import escape
from agent_config import AGENT_NAME, USER_NAME, USER_PASSWORD, personal_commands, agent_knowledge
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
from sidekick_tools import playwright_tools, other_tools
import uuid
import os
import asyncio
from datetime import datetime

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
            self.tools += additional_tools
            print(f"✓ Loaded {len(self.tools)} tools successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load tools: {e}")
            self.tools = []

        try:
            worker_llm = ChatGroq(
                model="deepseek-r1-distill-llama-70b",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.7,
                max_tokens=4096,
                stop_sequences=["<function=", "</function>", "function="]
            )

            if self.tools:
                self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)
            else:
                self.worker_llm_with_tools = worker_llm

            evaluator_llm = ChatGroq(
                model="deepseek-r1-distill-llama-70b",
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
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            success_criteria = state.get("success_criteria", "Provide helpful response")
            tool_names = ", ".join([tool.name for tool in self.tools]) if self.tools else "None"

            system_message = (
              f"""You are {AGENT_NAME}, a personal AI assistant.
You assist {USER_NAME} with daily tasks, reminders, and knowledge.
Here is your background knowledge: {agent_knowledge}
Always respond in a friendly and helpful way.

CRITICAL INSTRUCTIONS:
- You have access to these tools: """
                + tool_names
                + """
- When you need to use a tool, the system will handle the tool calls automatically
- DO NOT write function calls in your response text like <function=...> or function=...
- DO NOT include any XML-like syntax or manual function calling in your responses
- Simply describe what you want to do in natural language, and if tools are needed, they will be called automatically
- Always respond in natural, conversational language

Available tools: """
                + tool_names
                + """

EXAMPLES of what NOT to do:
❌ WRONG: <function=search{"query":"weather"}></function>
❌ WRONG: function=search{"query":"weather"}
❌ WRONG: Let me search for that <function call syntax>

EXAMPLES of what TO do:
✅ CORRECT: I'll search for the weather information for you.
✅ CORRECT: Let me look that up for you.
✅ CORRECT: I'll help you find that information.

Current date and time: """
                + current_time
                + """

Success criteria: """
                + success_criteria
                + """

Respond naturally and conversationally. The system will automatically handle any tool usage needed.


🚨 CRITICAL INSTRUCTIONS FOR CALENDAR OPERATIONS:
- When users ask to book/schedule/create meetings or events, you MUST use the "booking_meeting" tool
- When users ask to see meetings/appointments, you MUST use the "get_meetings" tool
- When users ask to reschedule meetings, you MUST use the "reschedule_meeting" tool
- NEVER say you've booked a meeting unless you actually called the booking_meeting tool
- NEVER provide fake meeting IDs or pretend to have done something

📅 EXAMPLES OF WHAT YOU MUST DO:
- User: "Book a meeting with Alex tomorrow at 2pm"
  YOU MUST: Call booking_meeting tool with proper parameters
- User: "Show me my meetings"  
  YOU MUST: Call get_meetings tool

Available tools: {tool_names}

Current date and time: {current_time}
Success criteria: {success_criteria}
"""
            )

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

    def route_based_on_evaluation(self, state: State) -> str:
        try:
            if state.get("success_criteria_met") or state.get("user_input_needed"):
                return "END"
            return "worker"
        except Exception as e:
            print(f"Evaluation routing error: {e}")
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

    async def build_graph(self):
        """
        Build the LangGraph workflow graph
        """
        try:
            graph_builder = StateGraph(State)

            # Add nodes
            graph_builder.add_node("worker", self.worker)
            graph_builder.add_node("evaluator", self.evaluator)

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
                graph_builder.add_edge("tools", "worker")
            else:
                # If no tools, go directly to evaluator
                graph_builder.add_edge("worker", "evaluator")

            # Add conditional edges from evaluator
            graph_builder.add_conditional_edges(
                "evaluator", 
                self.route_based_on_evaluation, 
                {"worker": "worker", "END": END}
            )
            
            # Set start edge
            graph_builder.add_edge(START, "worker")

            # Compile the graph with memory
            self.graph = graph_builder.compile(checkpointer=self.memory)

        except Exception as e:
            print(f"✗ Graph building error: {e}")
            raise

    async def run_superstep(
            self, message: str, success_criteria: str, history: List[Dict], _allowed_tools: Optional[List[str]] = None
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

            if len(messages) >= 2:
                assistant_content = getattr(messages[-2], "content", "No response")
                feedback_content = getattr(messages[-1], "content", "No feedback")

                assistant_msg = {"role": "assistant", "content": assistant_content}
                feedback_msg = {"role": "system", "content": feedback_content}

                return history + [user_msg, assistant_msg, feedback_msg]
            else:
                error_msg = {"role": "assistant", "content": "I encountered an issue processing your request."}
                return history + [user_msg, error_msg]

        except Exception as e:
            print(f"Run superstep error: {e}")
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


    def test_tools(self):
        """Test if tools are working"""
        response = self.worker_llm_with_tools.invoke([
        HumanMessage(content="Search for weather in Karachi")
       ])
        print(f"Test response: {response}")
        print(f"Test tool calls: {getattr(response, 'tool_calls', [])}")


              



 





    








