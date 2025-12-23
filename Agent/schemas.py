from pydantic import BaseModel, Field
from typing import List, Optional

class OrderItem(BaseModel):
    product_name: str = Field(..., description="Name of the product ordered")
    quantity: int = Field(..., description="Quantity of the product")
    unit: str = Field(..., description="Unit of the product (e.g., box, piece, kg)")

class Order(BaseModel):
    items: List[OrderItem] = Field(..., description="List of items in the order")
    customer_name: Optional[str] = Field(None, description="Customer's name")
    contact_number: Optional[str] = Field(None, description="Customer's contact number")
    delivery_address: Optional[str] = Field(None, description="Delivery address including details")
    desired_delivery_date: Optional[str] = Field(None, description="Desired delivery date using standard format (e.g. YYYY-MM-DD) or 'ASAP' or text description")
    special_requests: Optional[str] = Field(None, description="Any special requests or notes")

class AgentResponse(BaseModel):
    reply_to_user: str = Field(..., description="The message to respond to the user with. In Korean.")
    order_state: str = Field(..., description="Current state: 'gathering_info', 'awaiting_confirmation', 'finalized'")
    collected_order: Optional[Order] = Field(None, description="The current state of the order information collected so far")

