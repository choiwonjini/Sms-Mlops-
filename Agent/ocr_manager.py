from vertexai.generative_models import GenerativeModel, Part, Image
import json
import os
import base64
from config import settings
from google.cloud import aiplatform

class OCRManager:
    """
    Manages OCR tasks using Vertex AI (Gemini or DeepSeek Endpoint).
    Extracts payment information from receipt images.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.OCR_MODEL_NAME
        self.endpoint_id = settings.OCR_ENDPOINT_ID
        
        # Determine provider based on Model Name AND Endpoint ID
        # If model name explicitly says 'gemini', use Gemini even if Endpoint ID is present.
        self.use_endpoint = False
        if self.endpoint_id and not self.model_name.lower().startswith("gemini"):
            self.use_endpoint = True
            
        if self.use_endpoint:
            # Initialize Vertex AI for Endpoint
            aiplatform.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
            self.endpoint = aiplatform.Endpoint(self.endpoint_id)
            print(f"OCRManager initialized via Endpoint (DeepSeek): {self.endpoint_id}")
        else:
            # Initialize Gemini
            self.model = GenerativeModel(self.model_name)
            print(f"OCRManager initialized via GenerativeModel (Gemini): {self.model_name}")

    def reset(self):
        """Resets the OCR manager (stateless for now)."""
        if settings.DEBUG:
            print("[OCRManager] Resetting...")
        pass

    def analyze_payment_receipt(self, image_path: str) -> dict:
        """
        Analyzes a payment receipt image and extracts information.
        Dispatches to Endpoint (DeepSeek) or Gemini based on config.
        """
        if not os.path.exists(image_path):
            return {"error": "Image file not found."}

        if self.use_endpoint:
            return self._analyze_via_endpoint(image_path)
        else:
            return self._analyze_via_gemini(image_path)

    def _analyze_via_gemini(self, image_path: str) -> dict:
        try:
            image = Image.load_from_file(image_path)
            
            prompt = """
            Analyze this bank transfer receipt/screenshot.
            Extract the following information in strict JSON format:
            {
                "sender_name": "Name of sender (입금자명/보내는 분)",
                "sender_bank": "Sender's Bank (보내는 분 은행/출금 계좌)",
                "receiver_bank": "Receiver's Bank (받는 분 은행/입금은행)",
                "receiver_account": "Receiver's Account No (입금 계좌번호)",
                "receiver_owner": "Receiver's Name (입금 예금주)",
                "amount": "Transfer amount (입금 금액)", 
                "date": "Transfer date (YYYY-MM-DD)",
                "time": "Transfer time (HH:MM:SS)"
            }
            If a field is missing, use null.
            Return ONLY the JSON.
            """
            
            response = self.model.generate_content([image, prompt])
            return self._parse_json_response(response.text)
            
        except Exception as e:
            return {"error": f"Gemini OCR Analysis failed: {str(e)}"}

    def _analyze_via_endpoint(self, image_path: str) -> dict:
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
                # Ensure properly padded base64 if needed, though standard b64encode is usually fine
                encoded_image = base64.b64encode(image_bytes).decode("utf-8")

            # DeepSeek-VL / OpenAI-Compatible Payload for Vertex AI Endpoint
            # Note: Vertex AI Model Garden 'DeepSeek OCR' MaaS typically uses standard VLM containers.
            # We use the OpenAI Chat Completion format which is the standard for 3rd party models on Vertex.
            
            # DeepSeek OCR is optimized for Markdown. We explicitly request JSON.
            prompt_text = """
            You are an advanced OCR engine.
            Analyze this bank transfer receipt image.
            Extract payment details into the following JSON structure:
            {
                "sender_name": "Name of sender",
                "sender_bank": "Sender's Bank",
                "receiver_bank": "Receiver's Bank",
                "receiver_account": "Receiver's Account Number",
                "receiver_owner": "Receiver's Name",
                "amount": "Amount", 
                "date": "YYYY-MM-DD",
                "time": "HH:MM:SS"
            }
            Return ONLY valid JSON.
            """

            # Construct OpenAI-style Chat Completion payload
            instance = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.1
            }

            # Vertex AI expects 'instances' list
            prediction = self.endpoint.predict(instances=[instance])
            
            # Response parsing: 
            # Vertex AI vLLM/TGI containers usually return standard OpenAI response body OR a simplified one.
            # prediction.predictions is a list.
            if not prediction.predictions:
                return {"error": "No prediction returned from endpoint."}
            
            raw_output = prediction.predictions[0]
            
            # If output mimics OpenAI choices...
            # It might be a full string, or a dict like {'choices': [...]} or just the content string.
            # We try to extract text safely.
            result_text = ""
            if isinstance(raw_output, str):
                result_text = raw_output
            elif isinstance(raw_output, dict):
                # Try OpenAI format candidates
                if "choices" in raw_output and len(raw_output["choices"]) > 0:
                    choice = raw_output["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        result_text = choice["message"]["content"] # Chat completion
                    elif "text" in choice:
                        result_text = choice["text"] # Text completion
                elif "content" in raw_output:
                    result_text = raw_output["content"]
                # Fallback to string dump
                else:
                    result_text = str(raw_output)
            
            return self._parse_json_response(result_text)

        except Exception as e:
            return {"error": f"DeepSeek Endpoint OCR Analysis failed: {str(e)}"}

    def _parse_json_response(self, text: str) -> dict:
        try:
            text = text.replace("```json", "").replace("```", "").strip()
            # Safety cleanup if the model chats a bit before json
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end+1]
            return json.loads(text)
        except json.JSONDecodeError:
             return {"error": "Failed to parse JSON from OCR response", "raw_text": text}
