from pydantic import BaseModel, Field
from typing import Optional

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="pay_123456789")
    customer_id: str = Field(..., example="USR_123")
    client_canvas_hash: str = Field(..., example="hash123")
    app_set_id: str = Field(..., example="appset123")
    vpa_handle_hash: Optional[str] = Field(None, example="vpa123")
    delivery_pincode: str = Field(..., example="560001")
    address_quality_score: float = Field(..., example=3.5)
    payment_method: str = Field(..., example="COD")
    cart_value_inr: float = Field(..., example=2500.0)
    category_volatility: str = Field(..., example="High (Apparel)")
    checkout_velocity_seconds: float = Field(..., example=45.2)
    is_discount_applied: int = Field(..., example=1)
    customer_order_history_count: int = Field(..., example=5)

class PredictionResponse(BaseModel):
    risk_score: float = Field(..., ge=0.0, le=1.0)
    defense_action: str = Field(..., example="ALLOW_FREE_COD")
