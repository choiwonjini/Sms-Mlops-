
from agent_engine import TextOrderAgent
import json

def test_reset():
    print("--- Initializing Agent ---")
    agent = TextOrderAgent()
    
    # Simulate turn 1
    print("\n--- Simulating Order ---")
    agent.update_order_state(customer_name="Test User", items=[{"product_name": "Apple", "quantity": 1, "unit": "kg"}])
    
    print("Current State (Before Reset):")
    print(json.dumps(agent._current_order, indent=2, ensure_ascii=False))
    
    if agent._current_order["customer_name"] != "Test User":
        print("FAILED: State not updated")
        return

    # Reset
    print("\n--- Calling reset_state() ---")
    agent.reset_state()
    
    print("Current State (After Reset):")
    print(json.dumps(agent._current_order, indent=2, ensure_ascii=False))
    
    # Check
    if agent._current_order["customer_name"] is None and len(agent._current_order["items"]) == 0:
        print("\nSUCCESS: Memory cleared!")
    else:
        print("\nFAILED: Memory still exists!")
        
    if agent._chat_session is None:
        print("SUCCESS: Chat session cleared!")
    else:
        print("FAILED: Chat session still exists!")

if __name__ == "__main__":
    test_reset()
