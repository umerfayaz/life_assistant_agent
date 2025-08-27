# Test this in your terminal:
from calendar_integration import create_event
result = create_event(
    summary="Test Meeting with Alex",
    start_time="2025-08-27T09:00:00",  # 9 AM without timezone
    end_time="2025-08-27T10:00:00"     # 10 AM without timezone
)
print(result)