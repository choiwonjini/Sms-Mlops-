import vertexai
from vertexai.preview import reasoning_engines
from vertexai.generative_models import GenerativeModel, Tool
from typing import List, Optional, Dict, Any
import json
from config import settings
from datetime import datetime


class TextOrderAgent:
    """An agent that helps customers order fruit."""
    
    def __init__(self, project_id: str = None, location: str = None, model_name: str = None, guide_path: str = None):
        self.project_id = project_id or settings.GCP_PROJECT_ID
        self.location = location or settings.GCP_LOCATION
        self.model_name = model_name or settings.MODEL_NAME
        
        # Initialize Vertex AI
        if self.project_id and self.project_id != "your-gcp-project-id":
            vertexai.init(project=self.project_id, location=self.location)

        # Define Tools
        from vertexai.generative_models import Tool, FunctionDeclaration
        
        get_store_info_func = FunctionDeclaration(
            name="get_store_info",
            description="Returns the list of available fruits, prices, and ordering guide.",
            parameters={"type": "OBJECT", "properties": {}}
        )

        update_order_state_func = FunctionDeclaration(
            name="update_order_state",
            description="Updates the current order with new information.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "items": {
                        "type": "ARRAY",
                        "description": "List of items to add, e.g. [{'product_name': 'Apple', 'quantity': 1, 'unit': 'box'}]",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "product_name": {"type": "STRING"},
                                "quantity": {"type": "INTEGER"},
                                "unit": {"type": "STRING"}
                            }
                        }
                    },
                    "customer_name": {"type": "STRING", "description": "Name of the customer"},
                    "contact_number": {"type": "STRING", "description": "Phone number"},
                    "delivery_address": {"type": "STRING", "description": "Delivery address"},
                    "desired_delivery_date": {"type": "STRING", "description": "Desired delivery date"},
                    "special_requests": {"type": "STRING", "description": "Any special requests"}
                }
            }
        )

        get_current_order_func = FunctionDeclaration(
            name="get_current_order",
            description="Returns the current state of the order.",
            parameters={"type": "OBJECT", "properties": {}}
        )

        verify_payment_func = FunctionDeclaration(
            name="verify_payment",
            description="Verifies payment by analyzing a receipt image from the transfer_image folder.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "image_name": {
                        "type": "STRING", 
                        "description": "Name of the image file to check (e.g., 'receipt.jpg'). If user didn't specify, Agent checks the directory."
                    }
                }
            }
        )

        finalize_order_func = FunctionDeclaration(
            name="finalize_order",
            description="Finalizes the order and returns the payment information. Only call after confirmation.",
            parameters={"type": "OBJECT", "properties": {}}
        )

        self.order_guide_tool = Tool(
            function_declarations=[
                get_store_info_func,
                update_order_state_func,
                get_current_order_func,
                finalize_order_func,
                verify_payment_func
            ]
        )
        
        # Internal State
        self._current_order: Dict[str, Any] = self._get_default_order_state()
        
        # Interaction State for Deterministic Flow
        # "ORDERING", "AWAITING_PAYMENT_PROOF", "AWAITING_SELLER_APPROVAL"
        self.interaction_state = "ORDERING"
        
        # Initialize OCR
        from ocr_manager import OCRManager
        self.ocr_manager = OCRManager(model_name=settings.OCR_MODEL_NAME)
        
        # Initialize Price Verifier
        from price_verifier import PriceVerifier
        self.price_verifier = PriceVerifier()
        
        # Determine Guide Path
        self.guide_path = guide_path if guide_path else f"{settings.GUIDES_DIR}/order_guide.txt"
        
        # Load Store Guide for System Prompt context
        self._initialize_model()
        
    def _initialize_model(self):
        """Loads guides and creates/recreates the GenerativeModel with system instructions."""
        store_guide_text = "Store information unavailable."
        address_guide_text = "Address validation guide unavailable."
        try:
            with open(self.guide_path, "r", encoding="utf-8") as f:
                store_guide_text = f.read()
            with open(f"{settings.GUIDES_DIR}/address_guide.txt", "r", encoding="utf-8") as f:
                address_guide_text = f.read()
        except Exception:
            pass

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        self.model = GenerativeModel(
            self.model_name,
            system_instruction=[
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
                "10. **PAYMENT GUIDE**: After `finalize_order`, provide the Account Info and STOP. The system will handle verification automatically.",
                
                "RULES:",
                "- **LANGUAGE**: Always respond in polite Korean (존댓말), roughly matching the user's tone. NEVER switch to English unless the user speaks English.",
                "- **EXACT NAMES**: Generally use the Product Name from the Guide. HOWEVER, you **MUST** modify the name to resolve options. If Guide says 'Tea (A/B)', and user picks 'A', you MUST record 'Tea (A)'. Do NOT preserve the original '(A/B)'.",
                "- **PARENT PRODUCT MAPPING**: If the user orders a specific option (e.g. 'Small Set'), find the PARENT Product Name in the guide and resolve it. Example: User 'Small Set', Guide '[LocknLock]... - Small Set', Record '[LocknLock]... (Small Set)'.",
                "- **QUANTITY VS UNIT**: Carefully distinguish between Item Count and Unit Size. Example: 'Apple 5kg' means Quantity=1, Unit='5kg'. 'Two 5kg Apples' means Quantity=2, Unit='5kg'. Do NOT put the size (5) in quantity.",
                "- **NO REPEATS**: Do NOT ask for information that the user has already provided. Check your state dictionary before asking.",
                
                # ENHANCED RULES FOR UNSTRUCTURED GUIDES
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
                "- **ORDER DATE**: Record the 'order_date' as the 'CURRENT DATE/TIME' date (YYYY-MM-DD) when the order is initiated.",
                "8. **DELIVERY LOGIC**: 'desired_delivery_date' MUST be the date the USER explicitly requests (e.g. 'I need it by Dec 25th'). If the user does NOT explicitly ask for a specific date, set 'desired_delivery_date' to null. DO NOT infer the delivery date from the guide's 'shipping schedule' (e.g. 'orders before 2pm ship today'). That is the *estimated* delivery, not the *desired* one. If the user asks 'When will it arrive?', answer them based on the guide, but keep 'desired_delivery_date' as null. NEVER use the order recording date as the 'desired_delivery_date'.",                "- **UNIT PRICE**: When adding items, try to identify the 'unit_price' from the guide if possible. The system will verify it later.",
                "- **SEQUENTIAL PROCESSING**: NEVER call `finalize_order` and `verify_payment` in the same turn. The user CANNOT deposit without the account info.",

                "  - 'Can I buy 1 set of A?' -> Call `update_order_state(items=[{'product_name': 'Set A', ...}])`.",
                "  - 'My name is Kim, phone 010-1234, address Seoul.' -> Call `update_order_state(customer_name='Kim', contact_number='010-1234', delivery_address='Seoul')`.",
                "  - 'Kim / 010-1234 / Seoul' (Slash separated) -> You MUST parse this pattern and extract ALL 3 fields into `update_order_state`.",
                "  - 'Leave at door' -> Call `update_order_state(special_requests='Leave at door')`.",
                
                "CRITICAL RULES:",
                "0. **MULTIPLE INFO EXTRACTION**: If the user provides multiple pieces of information in one message (e.g. Name/Address/Phone together, or Slash separated), you MUST extract ALL of them in a single `update_order_state` call. Do not skip any field.",
                "1. **INSTANT CAPTURE**: As soon as the user mentions ANY order detail (Product, Quantity, Name, Phone, Address), you MUST call `update_order_state` IMMEDIATELY. Do not wait.",
                "2. **DO NOT JUST TALK**: Never simply repeat the order in text (e.g. 'I saved your name'). You MUST record it in the system using the tool. Text without Tool Call = FAILURE.",
                "3. **NUMBERED ITEMS**: If user says '1번' or 'No. 1', look up the product name in the STORE GUIDE and record the FULL Product Name.",
                "4. **ACCUMULATE**: If the user adds items (e.g. 'Also add 1 item X'), call `update_order_state` with the NEW items. The system will merge them.",
                "5. **SPECIAL REQUESTS**: Capture comments like 'Leave at door' in the `special_requests` field.",
                "6. **UNIT REQUIRED**: Always include the 'unit' field in items (e.g. 'box', 'ea', 'kg'). If not specified, infer it from the guide or default to '개'.",
                
                "SAFETY & PRIVACY:",
                "- **SAFETY FIRST**: Ensure your responses are helpful and completely safe. Avoid generating any content that could be interpreted as harmful, unsafe, or sexually explicit.",
                "- **DATA PURPOSE**: When asking for personal information (Name, Phone, Address), ALWAYS clarify that it is solely for 'Order Processing and Delivery'.",
                "- **ALWAYS RESPOND**: If a user's request is unclear, unsafe, or impossible, you MUST provide a polite explanation or request clarification. NEVER stop generating text or return an empty response.",
                
                "- Always be polite and helpful."
            ],
            tools=[self.order_guide_tool]
        )

    def _get_default_order_state(self) -> Dict[str, Any]:
        return {
            "items": [],
            "customer_name": None,
            "contact_number": None,
            "delivery_address": None,
            "desired_delivery_date": None,
            "special_requests": None,
            "payment_info": None
        }

    def reset_state(self):
        """Resets the agent's internal state and chat session."""
        self._current_order = self._get_default_order_state()
        self.interaction_state = "ORDERING"
        self._chat_session = None
        if hasattr(self, "price_verifier"):
            self.price_verifier.reset()
        if hasattr(self, "ocr_manager"):
            self.ocr_manager.reset()
        if settings.DEBUG:
            print("[Agent] Memory and state have been reset.")

    def update_guide(self, guide_path: str):
        """Updates the guide path and re-initializes the model with new instructions."""
        self.guide_path = guide_path
        self._initialize_model()
        if settings.DEBUG:
            print(f"[Agent] Guide updated to: {guide_path}")

    def verify_payment(self, image_name: str = None) -> str:
        """
        Verifies payment by analyzing a receipt image from the transfer_image folder.
        If image_name is not provided, it looks for the most recent file in the folder.
        """
        import os
        transfer_dir = "transfer_image"
        
        target_image = None
        if image_name:
            if os.path.exists(f"{transfer_dir}/{image_name}"):
                target_image = f"{transfer_dir}/{image_name}"
        else:
            # Find most recent file
            files = [f for f in os.listdir(transfer_dir) if os.path.isfile(os.path.join(transfer_dir, f))]
            if files:
                latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(transfer_dir, x)))
                target_image = f"{transfer_dir}/{latest_file}"
        
        if not target_image:
            return "No receipt image found in 'transfer_image' folder. Please upload the receipt."
            
        # Call OCR
        result = self.ocr_manager.analyze_payment_receipt(target_image)
        
        if "error" in result:
            return f"Error verifying payment: {result['error']}"
            
        # Update State
        self._current_order["payment_info"] = result
        
        # --- Comparison Logic ---
        # Retrieve Stored Expected Total (or recalculate if missing)
        expected_total = self._current_order.get("expected_amount")
        if expected_total is None:
            expected_total = self._calculate_expected_total()
            self._current_order["expected_amount"] = expected_total
        
        # Format Order Summary
        items = self._current_order.get("items", [])
        order_summary = ", ".join([f"{item.get('product_name')} {item.get('unit')} x{item.get('quantity')}" for item in items])
        
        from messages import get_system_message

        return (f"{get_system_message('VERIFICATION_HEADLINE')}\n"
                f"OCR Data (From Receipt):\n"
                f"  - Sender: {result.get('sender_name')} ({result.get('sender_bank', 'Unknown Bank')})\n"
                f"  - Amount: {result.get('amount')}\n"
                f"  - Receiver (Store): {result.get('receiver_bank')} {result.get('receiver_account', '')} ({result.get('receiver_owner', '')})\n"
                f"  - Date/Time: {result.get('date')} {result.get('time')}\n\n"
                f"Current Order Data:\n"
                f"  - Customer: {self._current_order.get('customer_name')}\n"
                f"  - Items: {order_summary}\n"
                f"  - Expected Amount: {expected_total:,}원\n"
                f"  - Expected Store Account: {settings.BANK_ACCOUNT_INFO}\n"
                f"\n"
                f"{get_system_message('SYSTEM_QUERY')} \n"
                f"{get_system_message('SELLER_INSTRUCTION')}")

    def _calculate_expected_total(self) -> int:
        """
        Calculates expected total using PriceVerifier Agent.
        """
        try:
            items = self._current_order.get("items", [])
            if not items:
                return 0
                
            store_guide = self.get_store_info()
            
            # Delegate to Price Verifier
            verification_result = self.price_verifier.verify_price(store_guide, items)
            
            # Extract final total
            final_total = verification_result.get("final_total", 0)
            
            # Update Item Details with PriceVerifier's enriched data (Unit Price, Correct Name)
            verified_items = verification_result.get("items", [])
            if verified_items:
                # Merge logic: We blindly trust PriceVerifier's item breakdown for the structure
                # This ensures 'unit_price' and 'subtotal' are exactly what the Verifier calculated.
                # It also normalizes product names if the verifier corrected them.
                self._current_order["items"] = verified_items
                
            # If Verifier failed to return items (e.g. error), we keep the Agent's original items
            # but unit_price might be missing.
            
            # if settings.DEBUG:
            #     print(f"[PriceVerifier] Result: {json.dumps(verification_result, ensure_ascii=False)}")
                
            return int(final_total)
            
        except Exception as e:
            if settings.DEBUG:
                print(f"[Debug] Price Calc Error: {e}")
            return 0

    def get_store_info(self) -> str:
        """Returns the list of available fruits, prices, and ordering guide."""
        try:
            with open(self.guide_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error loading store info: {e}"

    def update_order_state(
        self, 
        items: Optional[List[Dict[str, Any]]] = None,
        customer_name: Optional[str] = None,
        contact_number: Optional[str] = None,
        delivery_address: Optional[str] = None,
        desired_delivery_date: Optional[str] = None,
        special_requests: Optional[str] = None
    ) -> str:
        """
        Updates the current order with new information.
        
        Args:
            items: List of items, e.g., [{"product_name": "Apple", "quantity": 1, "unit": "box"}]
            customer_name: Name of the customer
            contact_number: Phone number
            delivery_address: Delivery address
            desired_delivery_date: Desired date
            special_requests: Any special requests
        """
        updates = []
        if items:
            # Append new items to existing list (Additive)
            self._current_order["items"].extend(items)
            updates.append(f"Added items: {items}")

        # For other fields, overwrite only if provided (Non-empty)
        # This prevents clearing fields if the LLM accidentally passes None/Empty
        if customer_name:
            self._current_order["customer_name"] = customer_name
            updates.append(f"Set name to: {customer_name}")
        if contact_number:
            self._current_order["contact_number"] = contact_number
            updates.append(f"Set contact to: {contact_number}")
        if delivery_address:
            self._current_order["delivery_address"] = delivery_address
            updates.append(f"Set address to: {delivery_address}")
        if desired_delivery_date:
            self._current_order["desired_delivery_date"] = desired_delivery_date
            updates.append(f"Set date to: {desired_delivery_date}")
        if special_requests:
            self._current_order["special_requests"] = special_requests
            updates.append(f"Set special requests to: {special_requests}")
            
        # Always set Order Date if not present
        if "order_date" not in self._current_order:
            self._current_order["order_date"] = datetime.now().strftime("%Y-%m-%d")
            
        # Real-time Price Update
        total = self._calculate_expected_total()
        self._current_order["expected_amount"] = total
        
        if settings.DEBUG:
            return f"Order updated. Current State: {json.dumps(self._current_order, ensure_ascii=False, indent=2)}"
        return "Order updated."

    def get_current_order(self) -> str:
        """Returns the current state of the order."""
        return json.dumps(self._current_order, ensure_ascii=False, indent=2)

    def finalize_order(self) -> str:
        """
        Finalizes the order and returns the payment information. 
        Only call this after the user has confirmed the order details.
        """
        # Calculate and Freeze Expected Amount
        total_amount = self._calculate_expected_total()
        self._current_order["expected_amount"] = total_amount
        
        # In a real app, strict validation would happen here
        msg = (f"주문이 확정되었습니다. 입금 계좌는 {settings.BANK_ACCOUNT_INFO} 입니다.\n"
                f"금액은 총 {total_amount:,}원 입니다.\n"
                f"입금 후 이체 확인증이나 캡처 이미지를 보내주세요.")
                
        # Transition State
        self.interaction_state = "AWAITING_PAYMENT_PROOF"
        
        return msg

    def query(self, message: str, history: List[str] = None):
        """
        Processes a user message using the Reasoning Engine pattern locally.
        Runs a chat session with tool use.
        """
        # --- Deterministic State Check ---
        if self.interaction_state == "AWAITING_PAYMENT_PROOF":
            # Simple heuristic: extract apparent filename or empty to default to latest.
            image_name = None
            words = message.split()
            for w in words:
                if w.lower().endswith('.png') or w.lower().endswith('.jpg'):
                    image_name = w
                    break
                    
            # Execute Verification Logic directly (skipping LLM)
            verification_result = self.verify_payment(image_name)
            
            # Use Extensible Message System for check
            from messages import get_system_message
            
            # If successful (headers match), transition to Approval
            if get_system_message('VERIFICATION_HEADLINE') in verification_result:
                self.interaction_state = "AWAITING_SELLER_APPROVAL"
                
            return verification_result

        elif self.interaction_state == "AWAITING_SELLER_APPROVAL":
            msg_lower = message.strip().lower()
            
            # Use Extensible Message System
            from messages import get_system_message
            
            if msg_lower in ["yes", "y", "예", "네"]:
                self.reset_state()
                return get_system_message("PAYMENT_CONFIRMED")
            elif msg_lower in ["no", "n", "아니오", "아니요"]:
                 self.reset_state()
                 return get_system_message("PAYMENT_REJECTED")
            else:
                return get_system_message("INVALID_INPUT")

        # If we want to persist the chat session across 'query' calls (multi-turn):
        if not hasattr(self, "_chat_session") or self._chat_session is None:
             self._chat_session = self.model.start_chat(response_validation=False)
        
        # Send message
        try:
             response = self._chat_session.send_message(message)
        except Exception as e:
             # Reset session if confirmed broken or just retry
             self._chat_session = self.model.start_chat(response_validation=False)
             response = self._chat_session.send_message(message)
        
        # Manual Tool Execution Loop
        max_turns = settings.MAX_TOOL_TURNS
        current_response = response
        
        try:
            for turn in range(max_turns):
                # 1. Candidate Check
                if not getattr(current_response, 'candidates', None) or len(current_response.candidates) == 0:
                    if settings.DEBUG:
                        print(f"  [Debug] No candidates in response.")
                    return "Error: No response from model (possibly blocked by safety filters)."
                
                candidate = current_response.candidates[0]
                
                # 2. Content Check
                if not hasattr(candidate, 'content') or candidate.content is None:
                    if settings.DEBUG:
                        print(f"  [Debug] Candidate has no content.")
                    return "Error: Model returned an empty candidate."

                # 3. Parts Check
                if not hasattr(candidate.content, 'parts') or not candidate.content.parts:
                    # Check for finish reason
                    finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
                    if settings.DEBUG:
                        print(f"  [Debug] No parts in candidate. Finish reason: {finish_reason}")
                    return f"Error: No content parts (Reason: {finish_reason})"

                # Iterate through parts to find function calls
                function_calls = []
                text_response = ""
                
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)
                    
                    try:
                        if part.text:
                            text_response += part.text
                    except Exception:
                        pass

                # If no function calls, return the text
                if not function_calls:
                    return text_response
                
                # Print thought trace if any
                if text_response and settings.DEBUG:
                    print(f"  [Agent Thought] {text_response}")

                # Execute all function calls found
                fc = function_calls[0]
                fn_name = fc.name
                fn_args = dict(fc.args)
                
                # Execute tool
                tool_result = "Unknown tool"
                
                if fn_name == "get_store_info":
                    tool_result = self.get_store_info(**fn_args)
                elif fn_name == "update_order_state":
                    tool_result = self.update_order_state(**fn_args)
                elif fn_name == "get_current_order":
                    tool_result = self.get_current_order(**fn_args)
                elif fn_name == "finalize_order":
                    tool_result = self.finalize_order(**fn_args)
                elif fn_name == "verify_payment":
                    tool_result = self.verify_payment(**fn_args)
                
                # Send result back to model
                from vertexai.generative_models import Part
                
                try:
                    current_response = self._chat_session.send_message(
                        Part.from_function_response(
                            name=fn_name,
                            response={"content": tool_result}
                        )
                    )
                except IndexError:
                    # SDK raises IndexError if the model returns no candidates (blocked)
                    if settings.DEBUG:
                        print("  [Debug] SDK Raised IndexError after tool result (Blocked).")
                    
                    # 2024-12-22: Fallback to Agent Thought if available
                    if text_response and text_response.strip():
                        if settings.DEBUG:
                            print("  [Fallback] Returning Agent Thought as response.")
                        return f"{text_response}\n(System: Internal action completed, but confirmation was muted.)"
                        
                    return "Error: Model response blocked after tool execution."
            
            return "Error: Maximum tool turns exceeded."

        except IndexError:
            # Catch IndexError in the main loop as well (for the initial response processing)
            if settings.DEBUG:
                print("  [Debug] SDK Raised IndexError during Turn loop.")
            return "Error: No response from model (SDK IndexError)."
        except Exception as e:
            if settings.DEBUG:
                import traceback
                print(f"[Critical Error] In query loop: {e}")
                traceback.print_exc()
            return f"Error: An error occurred during processing: {e}"
