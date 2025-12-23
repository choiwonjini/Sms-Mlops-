import pandas as pd
from test_agent import ImprovedTextOrderAgent
import os

def test_case_9():
    df = pd.read_csv("test_data/validation_synthetic_100.csv")
    row = df.iloc[8] # Case 9
    
    agent = ImprovedTextOrderAgent()
    
    guide_content = str(row['guide'])
    temp_guide_path = "guides/temp_guide_9_debug.txt"
    os.makedirs("guides", exist_ok=True)
    with open(temp_guide_path, "w", encoding="utf-8") as f:
        f.write(guide_content)
    
    agent.update_guide(guide_path=temp_guide_path)
    agent.reset_state()
    
    order_script = str(row['order']).split('\n')
    for turn in order_script:
        if not turn.strip(): continue
        print(f"User: {turn}")
        resp = agent.query(turn)
        print(f"Agent: {resp}")
        print(f"State: {agent.get_current_order()}")
        print("-" * 20)
    
    print("Final Mandatory Turn: 네")
    resp = agent.query("네")
    print(f"Agent: {resp}")
    print(f"Final State: {agent.get_current_order()}")

if __name__ == "__main__":
    test_case_9()
