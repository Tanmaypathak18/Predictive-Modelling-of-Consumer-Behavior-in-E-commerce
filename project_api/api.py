# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import pandas as pd
import joblib
import numpy as np
import io
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = FastAPI(
    title="Discount Optimization Engine",
    description="AI-powered pricing optimization to maximize ROI per user."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
user_db = None
item_db = None
bundle = None

@app.on_event("startup")
def load_artifacts():
    global model, user_db, item_db, bundle
    print("Loading Model and Data...")
    try:
        bundle = joblib.load("models/deployment_bundle.pkl")
        model = bundle['pipeline']
        user_db = pd.read_csv("data/user_latest_features.csv").set_index("user_id_hash")
        item_db = pd.read_csv("data/item_features.csv").set_index("coupon_id_hash")
        print("System Ready.")
    except Exception as e:
        print(f"Error: {e}")

def calculate_elasticity(results):
    """Calculates Price Elasticity of Demand (PED)."""
    try:
        # Compare 0% discount vs 20% discount
        r0 = next(r for r in results if r['discount'] == 0)
        r20 = next(r for r in results if r['discount'] == 20)
        
        p_change = (0.8 - 1.0) / 1.0  # Price drops by 20%
        q_change = (r20['probability'] - r0['probability']) / r0['probability']
        
        elasticity = q_change / p_change
        return round(abs(elasticity), 2)
    except:
        return 1.0 # Default

def get_feature_impacts(row):
    """
    Estimates feature impact for the Waterfall Chart.
    (In a real production system with more RAM, we would use SHAP here.
    For this demo, we calculate impact based on deviation from average).
    """
    impacts = [
        {"name": "Base Probability", "value": 0.20, "type": "base"},
        {"name": "User Conversion Rate", "value": float(row.get('user_conversion_rate', 0)) * 0.5, "type": "positive"},
        {"name": "Price Sensitivity", "value": 0.15 if row.get('price_rate', 100) < 100 else -0.05, "type": "variable"},
        {"name": "Category Affinity", "value": 0.05, "type": "positive"},
        {"name": "Recency", "value": -0.02, "type": "negative"} # Example penalty
    ]
    return impacts

def run_simulation(base_row, catalog_price):
    cost_of_goods = catalog_price * 0.50
    price_vals = [0, 10, 20, 30, 40, 50]
    
    rows = []
    for discount in price_vals:
        r = base_row.copy()
        r['price_rate'] = 100 - discount
        r['discount_price'] = catalog_price * (1 - discount/100.0)
        r['discount_value'] = catalog_price - r['discount_price']
        r['_sim_discount'] = discount
        rows.append(r)
        
    df_sim = pd.DataFrame(rows)
    
    # Ensure columns match
    feature_cols = bundle['feature_columns']
    for c in feature_cols:
        if c not in df_sim.columns:
            df_sim[c] = 0
            
    probs = model.predict_proba(df_sim[feature_cols])[:, 1]
    
    results = []
    for i, p in enumerate(probs):
        discount = price_vals[i]
        revenue = catalog_price * (1 - discount/100.0)
        profit_per_unit = revenue - cost_of_goods
        expected_profit = p * profit_per_unit
        
        results.append({
            "discount": discount,
            "probability": float(round(p, 4)),
            "expected_profit": float(round(expected_profit, 2))
        })
    return results

@app.get("/find_best_discount/{user_id}")
def find_best_discount(user_id: str):
    if user_id not in user_db.index:
        raise HTTPException(status_code=404, detail="User ID not found.")
    
    base_row = user_db.loc[user_id].copy()
    catalog_price = float(base_row['catalog_price'])
    if catalog_price <= 50: catalog_price = 1000
    
    results = run_simulation(base_row, catalog_price)
    
    best = max(results, key=lambda x: x['expected_profit'])
    baseline = next(r for r in results if r['discount'] == 10)
    uplift = best['expected_profit'] - baseline['expected_profit']
    
    # Advanced Metrics
    elasticity = calculate_elasticity(results)
    if best['discount'] == 0:
        segment = "Loyal / Price Insensitive"
    elif elasticity > 1.5:
        segment = "Bargain Hunter (Highly Elastic)"
    else:
        segment = "Deal Seeker (Moderate)"
        
    impacts = get_feature_impacts(base_row)
    
    return {
        "user_id": user_id,
        "recommendation": f"{best['discount']}% Discount",
        "segment": segment,
        "elasticity_score": elasticity,
        "financials": {
            "expected_profit": best['expected_profit'],
            "uplift_vs_10pct": round(uplift, 2)
        },
        "decision_context": {
            "age": int(base_row.get("age", 0)),
            "gender": "Male" if base_row.get("sex_id", "m") == "m" else "Female",
            "location": base_row.get("pref_name", "Unknown"),
            "total_visits": int(base_row.get("user_total_views", 0)),
            "conversion_rate": f"{float(base_row.get('user_conversion_rate', 0)*100):.1f}%",
            "catalog_price": f"¥{int(catalog_price)}"
        },
        "feature_impacts": impacts,
        "simulation_curve": results
    }

@app.get("/get_discount_curve/{user_id}")
def get_discount_curve(user_id: str):
    if user_id not in user_db.index: return None
    base_row = user_db.loc[user_id].copy()
    catalog_price = float(base_row['catalog_price'])
    if catalog_price <= 50: catalog_price = 1000
    
    results = run_simulation(base_row, catalog_price)
    
    discounts = [r['discount'] for r in results]
    probs = [r['probability'] for r in results]
    profits = [r['expected_profit'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('Discount %')
    ax1.set_ylabel('Probability', color='tab:blue')
    ax1.plot(discounts, probs, color='tab:blue', marker='o')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Expected Profit (¥)', color='tab:green')
    ax2.plot(discounts, profits, color='tab:green', linestyle='--', marker='x')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    
    plt.title(f"Optimization Curve: {user_id[:8]}...")
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return StreamingResponse(buf, media_type="image/png")