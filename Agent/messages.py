from config import settings

# Extensible Message Dictionary
# Users can add new languages here (e.g., 'japanese', 'chinese')
SYSTEM_MESSAGES = {
    "korean": {
        "PAYMENT_CONFIRMED": "결제가 확인되었습니다. 배송을 준비하겠습니다.",
        "PAYMENT_REJECTED": "결제가 거절되었습니다. 금액을 확인하고 이체 확인증을 다시 보내주세요.",
        "INVALID_INPUT": "잘못된 입력입니다. 결제를 승인하려면 '예', 거절하려면 '아니오'를 입력해주세요.",
        "VERIFICATION_HEADLINE": "--- 결제 검증 요청 ---",
        "SYSTEM_QUERY": "시스템 질문: 영수증 금액이 주문 합계(계산됨)와 일치합니까?",
        "SELLER_INSTRUCTION": "판매자님, 승인하려면 '예', 거절하려면 '아니오'를 입력해 주세요."
    },
    "english": {
        "PAYMENT_CONFIRMED": "Payment confirmed. Processing delivery.",
        "PAYMENT_REJECTED": "Payment rejected. Please check the amount and send receipt again.",
        "INVALID_INPUT": "Invalid input. Please type 'Yes' to confirm payment or 'No' to reject.",
        "VERIFICATION_HEADLINE": "--- Payment Verification Required ---",
        "SYSTEM_QUERY": "SYSTEM QUERY: Does the receipt amount match the order total?",
        "SELLER_INSTRUCTION": "Seller, please type 'Yes' to confirm payment or 'No' to reject."
    }
}

def get_system_message(msg_key: str) -> str:
    """
    Retrieves the message for the current configured language.
    Falls back to English if language is not supported.
    """
    current_lang = settings.LANGUAGE.lower()
    
    # Fallback to english if language not found
    lang_messages = SYSTEM_MESSAGES.get(current_lang, SYSTEM_MESSAGES["english"])
    
    return lang_messages.get(msg_key, f"[Missing Message: {msg_key}]")
