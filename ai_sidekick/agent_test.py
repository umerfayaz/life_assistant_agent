# Save this as test_agent_tools.py and run it

import asyncio
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

async def test_agent_calendar_directly():
    """Test if the agent LLM actually calls calendar tools"""
    
    print("🧪 Testing Agent LLM Tool Calling Directly")
    print("=" * 50)
    
    try:
        # Import your tools
        from sidekick_tools import other_tools
        
        # Get tools
        tools = await other_tools()
        calendar_tools = [t for t in tools if t.name == "booking_meeting"]
        
        if not calendar_tools:
            print("❌ No booking_meeting tool found!")
            return
            
        print(f"✅ Found booking_meeting tool: {calendar_tools[0].name}")
        
        # Create LLM with tools (same as your agent)
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=4096,
        )
        
        llm_with_tools = llm.bind_tools(calendar_tools)
        
        # Test with the same system prompt style as your agent
        system_prompt = """You are Mindara, a personal AI assistant.

🚨 CRITICAL INSTRUCTIONS:
- When users ask to book meetings, you MUST use the "booking_meeting" tool
- NEVER say you've booked a meeting unless you actually called the tool
- NEVER provide fake responses

Available tools: booking_meeting

Use the tools when appropriate!"""
        
        user_request = "Book a meeting called 'Agent Test Meeting' for tomorrow at 10am to 11am. Use start_time='2025-08-27T10:00:00' and end_time='2025-08-27T11:00:00'"
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_request)
        ]
        
        print(f"📤 Sending request: {user_request}")
        
        response = llm_with_tools.invoke(messages)
        
        print(f"📨 Response type: {type(response)}")
        print(f"📨 Response content: {getattr(response, 'content', 'No content')}")
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print("✅ SUCCESS: LLM made tool calls!")
            print(f"📞 Tool calls: {response.tool_calls}")
            
            # Actually execute the tool call
            tool = calendar_tools[0]
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'booking_meeting':
                    print(f"🔧 Executing tool with args: {tool_call['args']}")
                    result = tool.invoke(tool_call['args'])
                    print(f"✅ Tool execution result: {result}")
                    
                    if "created successfully" in result:
                        print("🎉 SUCCESS: Meeting actually created in Google Calendar!")
                        return True
                    else:
                        print("❌ Tool executed but no meeting created")
                        return False
        else:
            print("❌ PROBLEM: LLM did NOT make tool calls!")
            print("💡 The LLM is responding with text instead of using tools")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = asyncio.run(test_agent_calendar_directly())
    
    if success:
        print("\n🎯 DIAGNOSIS: Your tools work fine!")
        print("📋 PROBLEM: Your agent's worker function is interfering")
        print("📋 SOLUTION: Remove manual calendar handling from worker function")
    else:
        print("\n🎯 DIAGNOSIS: LLM tool calling issues")
        print("📋 CHECK: System prompt, tool binding, or LLM configuration")