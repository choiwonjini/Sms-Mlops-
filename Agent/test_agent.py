


import pandas as pd
import os
import json
import ast
from datetime import datetime
import sys
import time
import re
from unittest.mock import MagicMock

# Create a Dummy OCR Manager to prevent import errors if libs are missing
# and to avoid MagicMock serialization issues.
class DummyOCRManager:
    def __init__(self, model_name=None):
        pass
    def analyze_payment_receipt(self, image_path):
        return {"error": "OCR Disabled in Test Mode"}
    def reset(self):
        pass

# Mock the module so agent_engine imports this Dummy class
mock_ocr_module = MagicMock()
mock_ocr_module.OCRManager = DummyOCRManager
sys.modules["ocr_manager"] = mock_ocr_module

from agent_engine import TextOrderAgent
from config import settings

INPUT_DELAY = 2.0
ERROR_DELAY = 2.0
MAX_RETRIES = 5

def normalize_string(s: str) -> str:
    """Removes spaces, punctuation, mixed case for looses comparison."""
    if pd.isna(s) or s is None:
        return ""
    # Remove special chars but keep alphanum and Korean
    s = str(s).lower()
    return re.sub(r'[^a-zA-Z0-9가-힣]', '', s)

def is_loose_match(pred: str, label: str) -> bool:
    """Checks if normalized strings match or are contained in each other."""
    n_pred = normalize_string(pred)
    n_label = normalize_string(label)
    if not n_pred or not n_label:
        return n_pred == n_label
    return n_pred == n_label or n_pred in n_label or n_label in n_pred

def calculate_correctness(predict_str: str, label_str: str) -> float:
    """
    Compares prediction info with label info to calculate correctness score (0.0 to 1.0).
    Uses loose matching (normalization/substring) for better robustness without LLM.
    """
    try:
        # 1. Parse JSON / Dict String
        if not label_str or pd.isna(label_str):
            return 0.0
            
        if isinstance(predict_str, dict):
            pred = predict_str
        else:
            try:
                pred = json.loads(predict_str)
            except:
                pred = {}
             
        # Try to clean label_str for eval
        # Handle unquoted parenthesis groups: key: (value), or key: (value)}
        clean_label = re.sub(r':\s*(\([^)]+\))([,}])', r': "\1"\2', label_str)
        
        eval_ctx = {
            "null": None, 
            "true": True, 
            "false": False,
            "nan": None
        }
        
        try:
            label = eval(clean_label, {}, eval_ctx)
        except Exception as e:
            try:
                clean_json = clean_label.replace("null", "null").replace("True", "true").replace("False", "false")
                label = json.loads(clean_json)
            except:
                # Fallback: Regex Extraction
                label = {}
                # Extract simple fields
                for k in ["customer_name", "contact_number", "delivery_address", "desired_delivery_date", "special_requests"]:
                    m = re.search(f"['\"]?{k}['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", label_str)
                    if m: label[k] = m.group(1).strip()
                
                # Extract items (Rough approximation)
                # Look for product_name
                items = []
                # Simple pattern: 'product_name': 'X', 'quantity': 1
                # This is hard to do perfectly with regex, but let's try finding all products
                prods = re.findall(r"['\"]?product_name['\"]?\s*[:=]\s*['\"]([^'\"]+)['\"]", label_str)
                qtys  = re.findall(r"['\"]?quantity['\"]?\s*[:=]\s*(\d+)", label_str)
                
                if prods:
                    for i, p in enumerate(prods):
                        q = qtys[i] if i < len(qtys) else 1
                        items.append({"product_name": p, "quantity": int(q)})
                    label["items"] = items
                    
                if not label:
                    return 0.0
            
        # Fields to compare
        core_fields = [
            "customer_name", 
            "contact_number", 
            "delivery_address", 
            "desired_delivery_date",
            "expected_amount"
        ]
        
        matches = 0
        total_checks = len(core_fields) + 1 # +1 for Items
        
        # 2. Compare Core Fields
        for field in core_fields:
            p_val = pred.get(field)
            l_val = label.get(field)
            
            # Special handling for Amount
            if field == "expected_amount":
                try:
                    p_num = int(str(p_val).replace(',', '').replace('원', ''))
                    l_num = int(str(l_val).replace(',', '').replace('원', ''))
                    if p_num == l_num:
                        matches += 1
                        continue
                except:
                    pass 
            
            # Loose String Matching
            if is_loose_match(p_val, l_val):
                matches += 1
            else:
                pass
                
        # 3. Compare Items
        pred_items = pred.get("items", [])
        label_items = label.get("items", [])
        
        items_score = 0
        if not label_items:
            # If label has no items, but prediction implies items, it's mismatch?
            # Or if label is empty and pred is empty -> 1.0
            if not pred_items:
                items_score = 1.0
            else:
                items_score = 0.0
        else:
            # Item Scoring (Rough Approximation)
            pred_items = pred.get("items", [])
            label_items = label.get("items", [])
            
            matched_indices = set()
            matches_count = 0
            
            for l_item in label_items:
                l_name = l_item.get("product_name", "")
                l_qty = str(l_item.get("quantity", ""))
                
                found_idx = -1
                for idx, p_item in enumerate(pred_items):
                    if idx in matched_indices:
                        continue
                        
                    p_name = p_item.get("product_name", "")
                    p_qty = str(p_item.get("quantity", ""))
                    
                    # Check Match
                    if normalize_string(p_qty) == normalize_string(l_qty):
                        if is_loose_match(p_name, l_name):
                            found_idx = idx
                            break
                            
                if found_idx != -1:
                    matches_count += 1
                    matched_indices.add(found_idx)
            
            items_score = matches_count / len(label_items) if label_items else 1.0
            
        matches += items_score
        
        final_score = matches / total_checks
        return round(final_score, 2)

    except Exception as e:
        print(f"Error calculating score: {e}")
        return 0.0
    except Exception as e:
        print(f"Error calculating score: {e}")
        return 0.0

        return 0.0

class DummyPriceVerifier:
    """Pass-through verifier that does nothing, effectively disabling price verification."""
    def reset(self):
        pass

    def verify_price(self, store_guide, items):
        # Calculate totals locally since PriceVerifier is disabled
        total = 0
        formatted_items = []
        
        for item in items:
            qty = item.get('quantity', 0)
            price = item.get('unit_price', 0)
            subtotal = qty * price
            
            # Update item structure
            new_item = item.copy()
            new_item['subtotal'] = subtotal
            formatted_items.append(new_item)
            
            total += subtotal
            
        return {
            "items": formatted_items,
            "total_price": total,
            "item_total": total,
            "shipping_fee": 0, # Assuming free shipping for dummy mode or handle logic if needed
            "final_total": total,
            "reasoning": "Price Verification Disabled (Calculated by Main Agent)"
        }

class ImprovedTextOrderAgent(TextOrderAgent):
    """
    Subclass of TextOrderAgent with Enhanced System Prompt and Debuggingcapabilities.
    Used for testing without modifying the core agent_engine.py.
    """
    """
    Subclass of TextOrderAgent with Enhanced System Prompt and Debuggingcapabilities.
    Used for testing without modifying the core agent_engine.py.
    """
    def __init__(self, project_id: str = None, location: str = None, model_name: str = None, guide_path: str = None, use_price_verifier: bool = True):
        self.use_price_verifier = use_price_verifier
        
        # Call super init to setup basic state and tools
        super().__init__(project_id, location, model_name, guide_path)
        
        # [Test Enhancements] Override Tool Definition to allow 'unit_price' extraction
        if not use_price_verifier:
            print("  [Config] Price Verifier DISABLED (Using Dummy)")
            self.price_verifier = DummyPriceVerifier()
            
            # Re-define tool with unit_price field (which is missing in agent_engine.py)
            from vertexai.generative_models import Tool, FunctionDeclaration
            
            # Reconstruct the function declaration with unit_price
            update_order_state_func = FunctionDeclaration(
                name="update_order_state",
                description="Updates the current order with new information.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "items": {
                            "type": "ARRAY",
                            "description": "List of items to add, e.g. [{'product_name': 'Apple', 'quantity': 1, 'unit': 'box', 'unit_price': 3000}]",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "product_name": {"type": "STRING"},
                                    "quantity": {"type": "INTEGER"},
                                    "unit": {"type": "STRING"},
                                    "unit_price": {"type": "INTEGER", "description": "Price per unit"}
                                }
                            }
                        },
                        "customer_name": {"type": "STRING"},
                        "contact_number": {"type": "STRING"},
                        "delivery_address": {"type": "STRING"},
                        "desired_delivery_date": {"type": "STRING"},
                        "special_requests": {"type": "STRING"}
                    }
                }
            )
            
            # Clone other functions from parent (hard to access directly, so strictly only need to rebuild the tool if I can access others)
            # Actually, I need to recreate the whole Tool object. 
            # I can try to access the original FunctionDeclarations from the existing tool, but the SDK objects might be opaque.
            # Safe bet: Re-declare all of them locally or import them? No, they are local vars in __init__ of agent_engine.
            # I must redefine all of them or find a way to patch just one.
            # Vertex AI Tool constructor takes a list of FunctionDeclaration.
            
            # Let's see if we can extract existing funcs from self.order_guide_tool._raw_tool.function_declarations (implementation detail?)
            # Alternatively, since this is a test agent, I can just redefine the minimal set I need or all of them.
            # Redefining all is safer.
            
            get_store_info_func = FunctionDeclaration(
                name="get_store_info", description="Returns fruit list/prices.", parameters={"type": "OBJECT", "properties": {}}
            )
            get_current_order_func = FunctionDeclaration(
                name="get_current_order", description="Returns current order.", parameters={"type": "OBJECT", "properties": {}}
            )
            finalize_order_func = FunctionDeclaration(
                name="finalize_order", description="Finalizes order.", parameters={"type": "OBJECT", "properties": {}}
            )
            verify_payment_func = FunctionDeclaration(
                name="verify_payment", description="Verifies payment receipt.",
                parameters={"type": "OBJECT", "properties": {"image_name": {"type": "STRING"}}}
            )
            
            self.order_guide_tool = Tool(
                function_declarations=[
                    get_store_info_func,
                    update_order_state_func, # The modified one
                    get_current_order_func,
                    finalize_order_func,
                    verify_payment_func
                ]
            )
            print("  [Config] Tool Schema Overridden (Added unit_price)")

    def _initialize_model(self):
        """Loads guides and creates/recreates the GenerativeModel with enhanced instructions."""
        store_guide_text = "Store information unavailable."
        address_guide_text = "Address validation guide unavailable."
        try:
            with open(self.guide_path, "r", encoding="utf-8") as f:
                store_guide_text = f.read()
            with open(f"{settings.GUIDES_DIR}/address_guide.txt", "r", encoding="utf-8") as f:
                address_guide_text = f.read()
        except Exception:
            pass
            
        from vertexai.generative_models import GenerativeModel
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Enhanced Instructions (Synced with agent_engine.py + Test Fixes)
        enhanced_instructions = [
            f"CURRENT DATE/TIME: {current_time}",
            "You are an expert Order Processing Agent for the store defined in the STORE GUIDE.",
            f"STORE GUIDE:\n{store_guide_text}",
            f"ADDRESS GUIDE:\n{address_guide_text}",
            "CRITICAL: You ALREADY KNOW the store products from the guide. Do not check store info again.",
            
            "PROCESS FLOW:",
            "1. **ANALYZE FIRST MENTION**: Check the very first user message. If it contains ANY order info (Product, Quantity, Date, Name, Phone, Address), call `update_order_state` IMMEDIATELY to record it.",
            "2. **QUESTION ONLY**: If user asks regarding product/price WITHOUT ordering info (e.g. 'How much?'), Answer the question AND explicitly ask: 'Would you like to place an order?'.",
            "3. **MIXED INTENT**: If the user asks a question AND provides order info (e.g. 'How much is apple? I want one'), FIRST answer the question, THEN record the order using `update_order_state`, and FINALLY ask for missing details.",
            "4. **GATHER**: Listen to the user. Save EVERY piece of info (Name, Phone, Address, Items, etc.) into your internal dictionary as soon as it is mentioned.",
            "5. **CHECK & CLARIFY**: If info is missing OR ambiguous (e.g. 'Apple' without size/type), ask for specific clarification. Do not guess.",
            "6. **VALIDATE ADDRESS**: Before confirming, strictly check the address against the ADDRESS GUIDE. If incomplete (e.g. only 'Seoul'), ask for details.",
            "7. **COMPLETENESS CHECK**: Do NOT ask 'Is this order correct?' or show the summary until you have ALL 5 required fields: Items, Name, Contact, Full Address, Delivery Date.",
            "8. **CONFIRM**: Once you have ALL info, naturally summarize the order and ask if it is correct. Do NOT instruct the user on exactly what words to say (e.g. avoid 'Say Yes to confirm').",
            "9. **FINALIZE**: ONLY after the user confirms, call `finalize_order`.",
            
            "RULES:",
            "- **LANGUAGE**: Always respond in polite Korean (존댓말), roughly matching the user's tone. NEVER switch to English unless the user speaks English.",
            "- **EXACT NAMES**: Generally use the Product Name from the Guide. HOWEVER, you **MUST** modify the name to resolve options. If Guide says 'Tea (A/B)', and user picks 'A', you MUST record 'Tea (A)'. Do NOT preserve the original '(A/B)'.",
            "- **PARENT PRODUCT MAPPING**: If the user orders a specific option (e.g. 'Small Set'), find the PARENT Product Name in the guide and resolve it. Example: User 'Small Set', Guide '[LocknLock]... - Small Set', Record '[LocknLock]... (Small Set)'.",
            "- **QUANTITY VS UNIT**: Carefully distinguish between Item Count and Unit Size. Example: 'Apple 5kg' means Quantity=1, Unit='5kg'. 'Two 5kg Apples' means Quantity=2, Unit='5kg'. Do NOT put the size (5) in quantity.",
            "- **NO REPEATS**: Do NOT ask for information that the user has already provided. Check your state dictionary before asking.",
            
            # ENHANCED RULES FOR UNSTRUCTURED GUIDES (From agent_engine.py)
            "- **NUMBERED ITEMS**: If the Product Name in the Guide starts with a number (e.g. '1번 수제차', '2. Bagle'), YOU MUST KEEP that number in the recorded Product Name. Do NOT strip it. Example: User 'No. 1', Guide '1. Apple' -> Record '1. Apple'.",
            "- **MANDATORY OPTIONS**: If the Store Guide Product Name includes lists of options in parentheses (e.g. '(S/M/L)', '(Red/Blue)'), you MUST NOT `update_order_state` until the user specifies them. ASK 'Which color/size?' first.",
            "- **OPTION FORMATTING**: You MUST perform text replacement on the FULL Product Name. Replace ALL occurrences of `(A/B)` with selection `(A)`. Example: Guide 'Mix (Grapes/Apple) 500g (Grapes/Apple)', User 'Apple' -> Record 'Mix (Apple) 500g (Apple)'. DO NOT leave the group (A/B) in the string.",
            "- **UNSTRUCTURED GUIDES**: If the guide lists items in sentences (e.g. 'Selling Kohlrabi for 900 won each'), identify 'Kohlrabi' as the Product and '900 won' as the price.",
            "- **CRITICAL**: If the user says 'Apple 5kg please' in the first turn, your FIRST action must be `update_order_state` with that item.",
            "- **ADDRESS VALIDATION**: Reject incomplete addresses like 'Gangnam', 'Seoul', 'My House'. Ask for specific details (City/Road/Number) in Korean.",
            "- **AMBIGUITY**: If user says 'Give me apples', ASK 'Which type? Home or Gift? 5kg or 10kg?'. Do not default.",
            "7. **COMPLETENESS CHECK**: Do NOT ask 'Is this order correct?' or show the summary until you have all required fields: Items, Name, Contact, Address. (Delivery Date determines completeness only if NOT fixed by guide).",
            "8. **CONFIRM**: Once you have ALL info, naturally summarize the order. CRITICAL: You MUST explicitly mention the 'Delivery Date' or 'Delivery Schedule' in your summary. Then ask if the order is correct.",
            "9. **FINALIZE**: ONLY after the user confirms, call `finalize_order`.",
            "- DATE FORMATTING: If the user says 'tomorrow' or 'next week', you MUST calculate the actual YYYY-MM-DD date based on 'CURRENT DATE/TIME' and store the specific date string.",
            "- **ORDER DATE**: Record the 'order_date' as the 'CURRENT DATE/TIME' date (YYYY-MM-DD) when the order is initiated.",
            "8. **DELIVERY LOGIC**:",
            "- 'desired_delivery_date' MUST be the date the USER explicitly requests (e.g. \"I need it by Dec 25th\").",
            "- If the user does NOT explicitly ask for a specific date, set 'desired_delivery_date' to null.",
            "- DO NOT infer the delivery date from the guide's \"shipping schedule\" (e.g. \"orders before 2pm ship today\"). That is the *estimated* delivery, not the *desired* one.",
            "- If the user asks \"When will it arrive?\", answer them based on the guide, but keep 'desired_delivery_date' as null.",
            "- NEVER use the order recording date as the 'desired_delivery_date'.",
            "- **UNIT PRICE**: When adding items, try to identify the 'unit_price' from the guide if possible. The system will verify it later.",
            # OCR Rules removed
            
            "SAFETY & PRIVACY:",
            "- **SAFETY FIRST**: Ensure your responses are helpful and completely safe. Avoid generating any content that could be interpreted as harmful, unsafe, or sexually explicit.",
            "- **DATA PURPOSE**: When asking for personal information (Name, Phone, Address), ALWAYS clarify that it is solely for 'Order Processing and Delivery'.",
            "- **ALWAYS RESPOND**: If a user's request is unclear, unsafe, or impossible, you MUST provide a polite explanation or request clarification. NEVER stop generating text or return an empty response.",
            
            "- Always be polite and helpful.",
            
            # 2. TEST-SPECIFIC ENHANCEMENTS (To solve current failures)
            "CRITICAL RULES FOR STATE UPDATES:",
            "0. **MULTIPLE INFO EXTRACTION**: If the user provides multiple pieces of information in one message (e.g. Name/Address/Phone together, or Slash separated), you MUST extract ALL of them in a single `update_order_state` call. Do not skip any field.",
            "1. **INSTANT CAPTURE**: As soon as the user mentions ANY order detail (Product, Quantity, Name, Phone, Address), you MUST call `update_order_state` IMMEDIATELY. Do not wait.",
            "2. **DO NOT JUST TALK**: Never simply repeat the order in text (e.g. 'I saved your name'). You MUST record it in the system using the tool. Text without Tool Call = FAILURE.",
            "3. **NUMBERED ITEMS**: If user says '1번' or 'No. 1', look up the product name in the STORE GUIDE and record the FULL Product Name.",
            "4. **ACCUMULATE**: If the user adds items (e.g. 'Also add 1 item X'), call `update_order_state` with the NEW items. The system will merge them.",
            "5. **SPECIAL REQUESTS**: Capture comments like 'Leave at door' in `special_requests` field.",
            "6. **UNIT REQUIRED**: Always include the 'unit' field in items (e.g. 'box', 'ea', 'kg'). If not specified, infer it from the guide or default to '개'.",
        ]
        
        # 3. CONDITIONAL RULES (If PriceVerifier is DISABLED, Agent takes full responsibility)
        if not self.use_price_verifier:
            enhanced_instructions.extend([
                "PRICE VERIFIER DISABLED - STRICT FORMATTING REQUIRED:",
                "Because the verification step is skipped, YOU must format Product Names PERFECTLY.",
                "1. **STRICT OPTION FORMAT**: Convert 'Product (A/B)' to 'Product (A)' immediately.",
                "2. **NO AMBIGUITY**: Do not just record 'Set'. Record 'Gift Set (Large)' exactly as shown in the guide (with option resolved).",
                "3. **FINAL NAMES**: The `product_name` you record will be final. Ensure it matches the Store Guide exactly (including spacing).",
                "4. **PRICE EXTRACTION (CRITICAL)**: You *must* look up the price in the guide and fill `unit_price`. Example: Check guide -> 'Apple 3000 won' -> call `update_order_state(..., unit_price=3000)`. DO NOT set it to 0 or null."
            ])
        
        self.model = GenerativeModel(
            self.model_name,
            system_instruction=enhanced_instructions,
            tools=[self.order_guide_tool]
        )
        
    def update_order_state(self, **kwargs):
        # Debug Override
        print(f"  [DEBUG] Tool Call detected: update_order_state({kwargs})")
        return super().update_order_state(**kwargs)

def run_tests(limit=None, use_price_verifier=True):
    print(f"DEBUG: run_tests called with limit={limit}, use_price_verifier={use_price_verifier}")
    input_file = "test_data/validation_synthetic_100.csv"

    if not os.path.exists(input_file):
        if os.path.exists("validation_data_temp.CSV"):
            input_file = "validation_data_temp.CSV"
        else:
            print(f"Error: {input_file} not found.")
            return

    print(f"Loading test data from {input_file}...")
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decode failed, trying CP949...")
        df = pd.read_csv(input_file, encoding='cp949')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if 'no' in df.columns:
        df = df.dropna(subset=['no'])
    
    if limit:
        print(f"Limiting execution to first {limit} cases.")
        df = df.head(limit)
    
    results = []
    print("Starting Test Loop...")
    
    os.makedirs("guides", exist_ok=True)
    os.makedirs("test_result", exist_ok=True)
    
    import glob
    existing_temps = glob.glob("guides/temp_guide_*.txt")
    for f in existing_temps:
        try: os.remove(f)
        except: pass
        
    # Initialize Agent ONCE (Optimization)
    # Initialize Agent ONCE (Optimization)
    agent = ImprovedTextOrderAgent(use_price_verifier=use_price_verifier)
        
    for index, row in df.iterrows():
        temp_guide_path = None
        try:
            case_no = row.get('no', index)
            print(f"Processing Test Case #{case_no}...")
            
            # Rate Limit Protection
            time.sleep(20) 
            
            # 1. Setup Guide
            guide_content = row.get('guide', '')
            if pd.isna(guide_content): guide_content = ""
            
            # Fix unicode line endings if any
            guide_content = str(guide_content).replace('\r\n', '\n')
            
            temp_guide_path = f"guides/temp_guide_{index}.txt"
            with open(temp_guide_path, "w", encoding="utf-8") as f:
                f.write(guide_content)
                
            # 2. Update Agent (instead of re-initializing)
            agent.update_guide(guide_path=temp_guide_path)
            agent.reset_state()
            
            # 3. Simulate
            raw_order = str(row['order'])
            turns = raw_order.split('\n')
            transcript = ""
            turn_count = 0
            
            for t in turns:
                t = t.strip()
                if not t: continue
                

                # Retry Logic for Specific Error
                resp = ""
                current_retry_delay = ERROR_DELAY
                
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        # Capture state before query to check for partial updates later
                        prev_state = agent.get_current_order()
                        
                        resp = agent.query(t)
                        
                        # Check for specific error strings from agent_engine
                        # 1. "Error: An error occurred..." (General)
                        # 2. "Error: Model response blocked..." (SDK IndexError after tool)
                        # 3. "Error: No response from model..." (SDK IndexError main loop)
                        if any(err in resp for err in [
                            "Error: An error occurred", 
                            "Error: Model response blocked", 
                            "Error: No response from model"
                        ]):
                            # Check if state actually updated despite the error (Partial Success)
                            curr_state = agent.get_current_order()
                            if prev_state != curr_state:
                                print(f"  [Partial Success] Agent State updated despite error ('{resp}'). Skipping retry to prevent duplication.")
                                # Append a system note to the response so transcript makes sense
                                resp += " (System: Action completed, but response blocked by safety filter.)"
                                break

                            if attempt < MAX_RETRIES:
                                print(f"  [Retry] Agent Error detected ('{resp}'). Retrying in {current_retry_delay}s... (Attempt {attempt+1}/{MAX_RETRIES})")
                                time.sleep(current_retry_delay)
                                current_retry_delay *= 2
                                continue
                            else:
                                print(f"  [Fail] Max retries reached for error.")
                        else:
                            # Success (or other response)
                            break
                            
                    except Exception as e:
                        print(f"  [Error] Exception during query: {e}")
                        if attempt < MAX_RETRIES:
                             print(f"  [Retry] Exception detected. Retrying in {current_retry_delay}s... (Attempt {attempt+1}/{MAX_RETRIES})")
                             time.sleep(current_retry_delay)
                             current_retry_delay *= 2
                        else:
                             print(f"  [Fail] Max retries reached for exception.")
                             resp = f"System Error: {e}"

                transcript += f"User: {t}\n"
                transcript += f"Agent: {resp}\n"
                turn_count += 1
                
                # Input Delay between turns
                time.sleep(INPUT_DELAY)
                
            # --- MANDATORY FINAL TURN: "네" ---
            final_confirmation = "네"
            resp = ""
            current_retry_delay = ERROR_DELAY
            for attempt in range(MAX_RETRIES + 1):
                try:
                    resp = agent.query(final_confirmation)
                    if any(err in resp for err in [
                        "Error: An error occurred", 
                        "Error: Model response blocked", 
                        "Error: No response from model"
                    ]):
                        if attempt < MAX_RETRIES:
                            print(f"  [Retry] Agent Error detected ('{resp}'). Retrying in {current_retry_delay}s... (Attempt {attempt+1}/{MAX_RETRIES})")
                            time.sleep(current_retry_delay)
                            current_retry_delay *= 2
                            continue
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        print(f"  [Retry] Exception detected during confirmation. Retrying in {current_retry_delay}s... (Attempt {attempt+1}/{MAX_RETRIES})")
                        time.sleep(current_retry_delay)
                        current_retry_delay *= 2
                    else:
                        resp = f"System Error: {e}"
            
            transcript += f"User: {final_confirmation}\n"
            transcript += f"Agent: {resp}\n"
            turn_count += 1
            # ----------------------------------
            final_state = agent.get_current_order()
            label = row.get('label', '{}')
            
            # Calculate item count from LABEL (as per user request)
            # Moved before calculate_correctness so it persists even if scoring crashes
            item_count = 0
            try:
                # Label is likely a JSON string from the CSV row
                label_data = label
                if isinstance(label_data, str):
                    try:
                        label_data = json.loads(label_data)
                    except Exception as e:
                        try:
                            # Fallback to AST for Python-like dict strings
                            label_data = ast.literal_eval(label_data)
                        except:
                            # Final Fallback: Regex Counting
                            # Count occurrences of "product_name" (any quote style)
                            match_count = len(re.findall(r'product_name', str(label)))
                            if match_count > 0:
                                item_count = match_count
                                # Skip the dict check below if we used regex
                                label_data = {} # Clear it
                            else:
                                label_data = {}
                
                if isinstance(label_data, dict):
                     # If we successfully parsed it, use the list length (more accurate)
                     # If regex was used, item_count is already set
                     if 'items' in label_data:
                        item_count = len(label_data.get('items', []))
            except:
                item_count = 0
            
            score = calculate_correctness(final_state, label)
            
            print(f"  -> Score: {score}")
            

            results.append({
                "no": case_no,
                "order": transcript.strip(),
                "turn": turn_count,
                "item_count": item_count,
                "predict": final_state,
                "label": label,
                "correct_score": score
            })
            
        except Exception as e:
            print(f"Error on Case #{index}: {e}")
            # Try to calculate item count even in error case if possible
            try:
                # Basic Regex Fallback for error cases
                item_count = len(re.findall(r'product_name', str(row.get('label', ''))))
            except:
                item_count = 0
                
            results.append({
                "no": row.get('no', index),
                "order": "ERROR",
                "turn": 0,
                "item_count": item_count,
                "predict": str(e),
                "label": row.get('label', ''),
                "correct_score": 0.0
            })
        finally:
            # Cleanup
            if temp_guide_path and os.path.exists(temp_guide_path):
                try:
                    os.remove(temp_guide_path)
                except Exception as e:
                    print(f"Warning: Failed to cleanup {temp_guide_path}: {e}")
            
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"test_result/test_result_{timestamp}.csv"
    
    res_df = pd.DataFrame(results)
    
    cols = ["no", "order", "turn", "item_count", "predict", "label", "correct_score"]
    for c in cols:
        if c not in res_df.columns: res_df[c] = ""
    res_df = res_df[cols]
    
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Test Completed. Results saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Agent Tests")
    parser.add_argument("--limit", type=int, default=None, help="Number of test cases to run")
    parser.add_argument("--no-price-verifier", action="store_true", help="Disable Price Verifier (use dummy)")
    args = parser.parse_args()
    
    # Invert flag: --no-price-verifier means use_price_verifier=False
    run_tests(limit=args.limit, use_price_verifier=not args.no_price_verifier)

