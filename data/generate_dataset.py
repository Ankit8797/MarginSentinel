import pandas as pd
import numpy as np
from faker import Faker
import hashlib
 
fake = Faker('en_IN')
Faker.seed(42)
np.random.seed(42)
 
NUM_LEGIT = 9000
NUM_POWER_USERS = 500          # innocent hard negatives (look ring-like but aren't)
NUM_FULL_RINGS = 35            # clean, obvious rings
NUM_CAMOUFLAGED_RINGS = 15     # partial-signal rings (harder to catch)
USERS_PER_RING = 10
 
 
def generate_hash(text: str) -> str:
    """Simulates secure, one-way hashes used by payment gateways."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]
 
 
def base_row(pay_method, cart_value, discount, address_q, velocity,
             category, is_rto, is_fraud_ring, ring_id, order_history,
             canvas=None, app_set=None, vpa=None):
    return {
        'transaction_id': f"pay_{fake.unique.random_number(digits=12)}",
        'customer_id': f"USR_{fake.unique.random_number(digits=6)}",
        'client_canvas_hash': canvas or generate_hash(fake.uuid4()),
        'app_set_id': app_set or generate_hash(fake.uuid4()),
        'vpa_handle_hash': vpa if vpa is not None else (
            generate_hash(fake.user_name() + "@okicici") if pay_method == 'UPI' else None
        ),
        'delivery_pincode': fake.postcode(),
        'address_quality_score': round(address_q, 1),
        'payment_method': pay_method,
        'cart_value_inr': round(cart_value, 2),
        'category_volatility': category,
        'checkout_velocity_seconds': round(velocity, 1),
        'is_discount_applied': discount,
        'customer_order_history_count': order_history,
        'ring_id': ring_id,                 # NULL for genuine legit/power-user rows
        'is_rto': is_rto,                   # legit/innocent RTO outcome
        'is_fraud_ring': is_fraud_ring,     # ground-truth abuse-ring label (TARGET)
    }
 
 
print("Building synthetic Indian e-commerce dataset...")
data = []
 
# ---------------------------------------------------------------
# 1. Genuine legitimate majority (low risk, independent customers)
# ---------------------------------------------------------------
for _ in range(NUM_LEGIT):
    pay_method = np.random.choice(['COD', 'UPI', 'CARD'], p=[0.5, 0.4, 0.1])
    # Fixed: single clean draw per branch, no double-dipping rand()
    if pay_method == 'COD':
        is_rto = 1 if np.random.rand() < 0.15 else 0
    else:
        is_rto = 1 if np.random.rand() < 0.02 else 0
 
    data.append(base_row(
        pay_method=pay_method,
        cart_value=np.random.uniform(200, 5000),
        discount=np.random.choice([0, 1], p=[0.7, 0.3]),
        address_q=np.random.uniform(2.5, 5.0),
        velocity=np.random.uniform(30, 300),
        category=np.random.choice(['High (Apparel)', 'Medium (Electronics)', 'Low (Groceries)']),
        is_rto=is_rto,
        is_fraud_ring=0,
        ring_id=None,
        order_history=np.random.poisson(4),
    ))
 
# ---------------------------------------------------------------
# 2. Innocent "power users" - HARD NEGATIVES
#    Look ring-like on 1-2 features but are genuine, loyal customers.
#    Critical for measuring false-positive cost honestly.
# ---------------------------------------------------------------
for _ in range(NUM_POWER_USERS):
    data.append(base_row(
        pay_method=np.random.choice(['UPI', 'CARD']),
        cart_value=np.random.uniform(1500, 6000),
        discount=1,                                   # loyal users chase every sale
        address_q=np.random.uniform(3.0, 5.0),         # good address (unlike real rings)
        velocity=np.random.uniform(8, 25),              # fast checkout: saved payment info
        category=np.random.choice(['High (Apparel)', 'Medium (Electronics)']),
        is_rto=1 if np.random.rand() < 0.02 else 0,
        is_fraud_ring=0,
        ring_id=None,
        order_history=np.random.poisson(25) + 10,       # long tenure -> high LTV, high FP cost
    ))
 
# ---------------------------------------------------------------
# 3. Full/obvious fraud rings (Sybil attack, all signals trip)
# ---------------------------------------------------------------
for ring_num in range(NUM_FULL_RINGS):
    ring_id = f"RING_{ring_num:03d}"
    shared_canvas = generate_hash(f"ring_{ring_id}_canvas")
    shared_app_set = generate_hash(f"ring_{ring_id}_appset")
    shared_vpas = [generate_hash(f"fraud_{ring_id}_{i}@okhdfcbank") for i in range(3)]
 
    for _ in range(USERS_PER_RING):
        data.append(base_row(
            pay_method='COD',
            cart_value=np.random.uniform(2000, 10000),
            discount=1,
            address_q=np.random.uniform(1.0, 3.0),
            velocity=np.random.uniform(5, 20),
            category='High (Apparel)',
            is_rto=1 if np.random.rand() < 0.90 else 0,
            is_fraud_ring=1,
            ring_id=ring_id,
            order_history=np.random.poisson(1),          # brand-new accounts
            canvas=shared_canvas,
            app_set=shared_app_set,
            vpa=np.random.choice(shared_vpas),
        ))
 
# ---------------------------------------------------------------
# 4. Camouflaged rings - only 2-3 of 5 signals trip.
#    These are what make precision/recall a real, non-trivial metric.
# ---------------------------------------------------------------
for ring_num in range(NUM_CAMOUFLAGED_RINGS):
    ring_id = f"RING_CAMO_{ring_num:03d}"
    shared_canvas = generate_hash(f"camo_{ring_id}_canvas")
    shared_app_set = generate_hash(f"camo_{ring_id}_appset")   # device still shared...
    shared_vpas = [generate_hash(f"camo_{ring_id}_{i}@okhdfcbank") for i in range(3)]
 
    for _ in range(USERS_PER_RING):
        data.append(base_row(
            pay_method=np.random.choice(['COD', 'UPI']),        # ...but payment method varied
            cart_value=np.random.uniform(500, 4000),             # lower, less suspicious cart size
            discount=np.random.choice([0, 1], p=[0.4, 0.6]),
            address_q=np.random.uniform(2.0, 4.0),               # closer to normal range
            velocity=np.random.uniform(20, 60),                  # not obviously bot-fast
            category=np.random.choice(['High (Apparel)', 'Medium (Electronics)']),
            is_rto=1 if np.random.rand() < 0.60 else 0,
            is_fraud_ring=1,
            ring_id=ring_id,
            order_history=np.random.poisson(2),
            canvas=shared_canvas,
            app_set=shared_app_set,
            # only rotate VPA, not always - some pay differently to blend in
            vpa=np.random.choice(shared_vpas + [None]) if np.random.rand() < 0.7 else None,
        ))
 
# ---------------------------------------------------------------
# Assemble, shuffle, save
# ---------------------------------------------------------------
df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('ecommerce_rto_dataset.csv', index=False)
 
print(f"Success! Saved {len(df)} rows to ecommerce_rto_dataset.csv")
print(f"  Legit (incl. power users): {(df['is_fraud_ring'] == 0).sum()}")
print(f"  Fraud-ring rows:           {(df['is_fraud_ring'] == 1).sum()}")
print(f"  Unique fraud rings:        {df['ring_id'].nunique()}")
print("\nIMPORTANT — when you build the model:")
print("  Split train/test by unique ring_id (GroupShuffleSplit), NOT by row,")
print("  so held-out rings are truly unseen. This is what makes your")
print("  precision/recall numbers honest rather than memorized.")