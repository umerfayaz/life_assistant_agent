import os
from dotenv import load_dotenv




load_dotenv()


AGENT_NAME = os.getenv("AGENT_NAME","Mindara")
USER_NAME = os.getenv("USER_NAME","Umer")
USER_PASSWORD =os.getenv("USER_PASSWORD","techworld")


personal_commands = {
    "schedue_meeting": "check calendar and schedule meeting",
    "daily_brief": "Give the daily summary",
    "favourite_music": "play the favorite playlist"
}



agent_knowledge = {
    "name": AGENT_NAME,
    "job": "I’m designed to handle and optimize daily work so my human can focus on the bigger picture. Think of me as a digital employee who never sleeps, My goal is simple: take care of the workload, so my human doesn’t have to. I’m not just a chatbot — I’m a work partner, assistant, and problem-solver, built to save time, reduce stress, and get things done.",
    "purpose": "Assist with daily tasks, reminders, and knowledge", 
}





