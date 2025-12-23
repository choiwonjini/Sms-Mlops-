
from vertexai.generative_models import GenerativeModel
import json
import re
from config import settings
from typing import List, Dict, Any

class PriceVerifier:
    """
    Independent Agent for verifying order prices against the store guide.
    Uses a dedicated LLM call to ensure accuracy.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.PRICE_MODEL_NAME
        self.model = GenerativeModel(model_name=self.model_name)
    
    def reset(self):
        """Resets the verifier (stateless for now, but provides standard interface)."""
        if settings.DEBUG:
            print("[PriceVerifier] Resetting...")
        pass

    def verify_price(self, store_guide: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates the exact price for the given items based on the store guide.
        Returns a dictionary with detailed pricing breakdown.
        """
        if not items:
            return {
                "total_price": 0,
                "item_total": 0,
                "shipping": 0,
                "breakdown": [],
                "reasoning": "No items provided."
            }

        prompt = (
            "You are a Strict Price Verification Auditor. Your goal is to calculate the EXACT total price for an order based on the provided Store Guide.\n"
            "Do not guess. Match product names and units exactly.\n\n"
            f"STORE GUIDE:\n{store_guide}\n\n"
            f"ORDER ITEMS:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
            "INSTRUCTIONS:\n"
            "INSTRUCTIONS:\n"
            "1. **PRODUCT NORMALIZATION**: For each item, find its EXACT match in the Store Guide.\n"
            "   - If the user selected an option (e.g. 'Size L'), the 'product_name' MUST be formatted as: 'Product Name (Option)'.\n"
            "   - Example: Guide 'T-Shirt (S/M/L)', User 'Large' -> Output 'product_name': 'T-Shirt (L)'.\n"
            "   - Example: Guide 'Apple 5kg', User 'Apple' -> Output 'product_name': 'Apple 5kg'.\n"
            "   - DO NOT include the full option list (e.g. '(S/M/L)') in the final name.\n"
            "2. **UNIT PRICE**: Extract the correct unit price for the specific option.\n"
            "3. **UNIT PRICE (CRITICAL)**: Extract the integer unit price. Do NOT prioritize calculating the total yourself. Get the Unit Price right.\n"
            "4. **SHIPPING**: Identify the shipping fee.\n"
            "5. **OUTPUT**: Return valid JSON ONLY.\n\n"
            "JSON OUTPUT FORMAT:\n"
            "{\n"
            "  \"items\": [\n"
            "    {\"product_name\": \"Cleaned Product Name (Option)\", \"unit\": \"...\", \"unit_price\": 1000, \"quantity\": 2}\n"
            "  ],\n"
            "  \"shipping_fee\": 0,\n"
            "  \"reasoning\": \"Brief explanation\"\n"
            "}"
        )

        try:
            response = self.model.generate_content(prompt)
            
            # Safety check: if no candidates, .text will raise IndexError
            if not response.candidates or len(response.candidates) == 0:
                print(f"[PriceVerifier] Warning: No candidates in response.")
                return {"error": "No response from model (possibly blocked)"}
                
            text = response.text.strip()
            # Clean up code blocks if present
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
            
            result = json.loads(text)
            
            # --- HYBRID CALCULATION LOGIC (Python Side) ---
            # Trust the LLM for 'unit_price', but recalculate all arithmetic via Python
            calculated_item_total = 0
            formatted_items = []
            
            for item in result.get("items", []):
                # Ensure we have numbers
                qty = int(item.get("quantity", 0))
                price = int(item.get("unit_price", 0))
                subtotal = qty * price
                
                # Update the item with calculated subtotal
                item["subtotal"] = subtotal
                formatted_items.append(item)
                
                calculated_item_total += subtotal
                
            # Trust LLM for shipping fee (as logic is complex to extract), or default to 0
            shipping = int(result.get("shipping_fee", 0))
            
            # Recalculate Final Total
            final_total = calculated_item_total + shipping
            
            # Overwrite the result with calculated values
            result["items"] = formatted_items
            result["item_total"] = calculated_item_total
            result["total_price"] = calculated_item_total # Alias
            result["final_total"] = final_total
            
            return result

        except Exception as e:
            print(f"[PriceVerifier] Error: {e}")
            return {
                "total_price": 0,
                "item_total": 0,
                "shipping_fee": 0,
                "error": str(e)
            }
