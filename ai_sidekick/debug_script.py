# Fixed debug script - save as debug_script.py
import asyncio
from calendar_integration import create_event, list_events

async def debug_calendar_integration():
    """Debug the calendar integration step by step"""
    
    print(" DEBUGGING CALENDAR INTEGRATION")
    print("=" * 50)
    
    # Test 1: Direct calendar integration
    print("\n1️⃣ Testing direct calendar integration...")
    try:
        result = create_event(
            summary="Debug Test Meeting",
            description="Testing direct calendar integration",
            start_time="2025-08-26T10:00:00",
            end_time="2025-08-26T11:00:00"
        )
        print(f" Direct calendar result: {result}")
        
        # Check if it contains success indicators
        if "created successfully" in result or "Event ID:" in result:
            direct_success = True
        else:
            direct_success = False
            
    except Exception as e:
        print(f" Direct calendar failed: {e}")
        return False
    
    # Test 2: New structured function
    print("\n2️⃣ Testing new structured calendar function...")
    try:
        from sidekick_tools import book_meeting_structured
        
        result = book_meeting_structured(
            summary="Structured Test Meeting",
            description="Testing new structured function",
            start_time="2025-08-26T14:00:00",
            end_time="2025-08-26T15:00:00"
        )
        print(f" Structured function result: {result}")
        
        # Check if it contains success indicators
        if "created successfully" in result or "Event ID:" in result:
            structured_success = True
        else:
            structured_success = False
            
    except ImportError as e:
        print(f" Structured function not found: {e}")
        print("💡 Make sure you updated sidekick_tools.py with the new functions")
        structured_success = False
    except Exception as e:
        print(f" Structured function failed: {e}")
        structured_success = False
    
    # Test 3: List current events
    print("\n3️⃣ Listing current events...")
    try:
        events = list_events(10)
        print(f" Current events: {events}")
        
        # Check if our test meetings appear
        events_success = ("Debug Test Meeting" in events or "Structured Test Meeting" in events)
        
    except Exception as e:
        print(f" Listing events failed: {e}")
        events_success = False
    
    print("\n" + "=" * 50)
    print(" DIAGNOSIS:")
    
    if direct_success and structured_success and events_success:
        print(" SUCCESS: Calendar integration is working!")
        return True
    elif not direct_success:
        print(" PROBLEM: Direct calendar integration is broken")
        print(" Check your Google Calendar credentials (client_secret.json)")
        return False
    elif not structured_success:
        print(" PROBLEM: New structured functions not working")
        print(" Make sure you updated sidekick_tools.py correctly")
        return False
    else:
        print(" Partial success - some issues detected")
        return False

def check_agent_tools():
    """Check if the agent has the calendar tools loaded"""
    try:
        from sidekick_tools import other_tools
        import asyncio
        
        print("\n4️⃣ Checking agent tools...")
        tools = asyncio.run(other_tools())
        
        calendar_tools = [t for t in tools if 'meeting' in t.name.lower() or 'calendar' in t.name.lower()]
        
        print(f" Total tools loaded: {len(tools)}")
        print(f" Calendar tools found: {len(calendar_tools)}")
        
        for tool in calendar_tools:
            print(f"   - {tool.name}: {tool.description[:100]}...")
            
        return len(calendar_tools) > 0
        
    except Exception as e:
        print(f" Tool checking failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def test_llm_tool_calling():
    """Test if the LLM can actually call tools"""
    print("\n5️⃣ Testing LLM tool calling...")
    
    try:
        from langchain_groq import ChatGroq
        from sidekick_tools import other_tools
        import os
        import asyncio
        
        # Get tools
        tools = asyncio.run(other_tools())
        calendar_tools = [t for t in tools if t.name == "booking_meeting"]
        
        if not calendar_tools:
            print(" No booking_meeting tool found!")
            return False
            
        # Create LLM with tools
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1
        )
        
        llm_with_tools = llm.bind_tools(calendar_tools)
        
        # Test message
        from langchain_core.messages import HumanMessage
        test_message = HumanMessage(content="Book a meeting called 'LLM Test Meeting' tomorrow from 2pm to 3pm")
        
        response = llm_with_tools.invoke([test_message])
        
        print(f" LLM Response type: {type(response)}")
        print(f" LLM Response content: {getattr(response, 'content', 'No content')}")
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(" LLM is making tool calls!")
            print(f" Tool calls: {response.tool_calls}")
            return True
        else:
            print(" LLM is NOT making tool calls!")
            print(" The LLM should automatically use tools when you ask to book meetings")
            return False
            
    except Exception as e:
        print(f" LLM tool calling test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False

def check_credentials():
    """Check if Google Calendar credentials are set up"""
    print("\n0️⃣ Checking credentials...")
    
    import os
    
    if os.path.exists("client_secret.json"):
        print(" client_secret.json found")
        return True
    else:
        print(" client_secret.json NOT found")
        print(" You need to:")
        print("   1. Go to Google Cloud Console")
        print("   2. Enable Google Calendar API") 
        print("   3. Create OAuth credentials")
        print("   4. Download and rename to 'client_secret.json'")
        print("   5. Place it in your project folder")
        return False

if __name__ == "__main__":
    # Run all debug tests
    print(" Starting comprehensive calendar debug...")
    
    # Check credentials first
    creds_ok = check_credentials()
    
    if not creds_ok:
        print("\n STOPPING: Fix credentials first!")
        exit(1)
    
    # Test calendar integration
    success = asyncio.run(debug_calendar_integration())
    
    if success:
        # Test agent tools
        tools_ok = check_agent_tools()
        
        if tools_ok:
            # Test LLM tool calling
            llm_ok = test_llm_tool_calling()
            
            if llm_ok:
                print("\n ALL TESTS PASSED!")
                print("Your calendar integration should work now!")
            else:
                print("\n PROBLEM FOUND: LLM is not calling tools properly!")
                print(" SOLUTIONS:")
                print("   1. Check your system prompt in the agent")
                print("   2. Make sure you removed the manual calendar handling")
                print("   3. Verify the tool descriptions are clear")
        else:
            print("\n PROBLEM FOUND: Calendar tools not loaded properly!")
    else:
        print("\n PROBLEM FOUND: Basic calendar integration is broken!")
        print(" Check your Google Calendar credentials and permissions")