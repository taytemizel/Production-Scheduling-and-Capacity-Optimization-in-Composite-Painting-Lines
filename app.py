import streamlit as st
import pandas as pd
import time

# Custom Modules
from db_connection import fetch_all_items, insert_new_item, delete_item
from regression_engine import train_best_subset_model, predict_processing_time

# ---------------------------------------------------------
# 1. SETUP & STYLING
# ---------------------------------------------------------
st.set_page_config(page_title="TUSAŞ AI Predictor", page_icon="✈️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    div.stButton > button {
        background-color: #E30613; color: white; border-radius: 5px; font-weight: bold; border: none;
    }
    div.stButton > button:hover { background-color: #b30000; color: white; }
    div[data-testid="stMetric"] {
        background-color: #1c1f26; padding: 15px; border-radius: 10px; border-left: 5px solid #E30613;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. LOAD DATA & AUTO-TRAIN MODEL
# ---------------------------------------------------------
if 'ml_model' not in st.session_state: 
    st.session_state['ml_model'] = None

# Fetch data directly from MySQL
df_items = fetch_all_items()

# Automatically train the model if data exists and no model is loaded
if not df_items.empty and st.session_state['ml_model'] is None:
    model_info, err = train_best_subset_model(df_items)
    if model_info: 
        st.session_state['ml_model'] = model_info
    elif err: 
        st.error(f"Training Failed: {err}")

# ---------------------------------------------------------
# 3. SIDEBAR (MODEL INTELLIGENCE)
# ---------------------------------------------------------
with st.sidebar:
    try: st.image("logo.png", use_container_width=True)
    except: st.markdown("### TUSAŞ AI")
    
    st.markdown("### 🧠 AI Model Intelligence")
    
    if st.session_state['ml_model']:
        m = st.session_state['ml_model']
        st.success("Model Status: Optimized")
        st.metric("Model Accuracy (Adj R²)", f"{m['adj_r2']:.4f}")
        
        st.write("**Significant Predictors (P < 0.10):**")
        st.info(", ".join([f.upper() for f in m['features']]))
        
        st.divider()
        if st.button("🔄 Retrain on New Data"):
            st.session_state['ml_model'] = None
            st.rerun()
    else:
        st.warning("Model Not Trained")

# ---------------------------------------------------------
# 4. MAIN INTERFACE
# ---------------------------------------------------------
st.title("✈️ Processing Time Predictor")

if df_items.empty:
    st.warning("⚠️ Database is empty. The model requires historical data to make predictions.")
else:
    tab1, tab2 = st.tabs(["🔮 Predict New Item", "📦 Database View"])

    # --- TAB 1: PREDICTION ---
    with tab1:
        c_form, c_res = st.columns([1, 1])
        
        with c_form:
            st.subheader("Enter Item Parameters")
            with st.form("pred_form"):
                c1, c2 = st.columns(2)
                # Auto-incrementing Part ID
                next_id = int(df_items['part_id'].max() + 1) if not df_items.empty else 1
                new_id = c1.number_input("Part ID", value=next_id, step=1)
                
                # Categorical Inputs (User sees text)
                comp_txt = c2.selectbox("Complexity", ["BASIC", "COMPLEX"])
                
                c3, c4 = st.columns(2)
                sz_txt = c3.selectbox("Size", ["SMALL", "MEDIUM", "LARGE", "VERY LARGE"])
                pnt = c4.number_input("Paint Layers", min_value=1, value=1)
                
                c5, c6, c7 = st.columns(3)
                wi = c5.number_input("Width (cm)", min_value=1.0, value=50.0)
                hi = c6.number_input("Height (cm)", min_value=1.0, value=50.0)
                li = c7.number_input("Length (cm)", min_value=1.0, value=10.0)
                
                predict_btn = st.form_submit_button("🔍 Predict Processing Time")
                
        with c_res:
            st.subheader("Prediction Result")
            
            if predict_btn and st.session_state['ml_model']:
                
                # Auto-calculate dummy variables in the background
                val_dbasic = 1 if comp_txt == "BASIC" else 0
                val_dsmall = 1 if sz_txt == "SMALL" else 0
                val_dmedium = 1 if sz_txt == "MEDIUM" else 0
                val_dlarge = 1 if sz_txt == "LARGE" else 0
                
                # Prepare data for Model
                model_input = {
                    'w_i': wi, 'h_i': hi, 'l_i': li, 'paint_used': pnt,
                    'dbasic': val_dbasic, 'dsmall': val_dsmall, 
                    'dmedium': val_dmedium, 'dlarge': val_dlarge
                }
                
                # Prepare data for Database (Includes text + dummies)
                full_record = model_input.copy()
                full_record.update({
                    'part_id': new_id,
                    'comp_level': comp_txt,
                    'size': sz_txt
                })
                
                # Execute Prediction
                pred_pi = predict_processing_time(st.session_state['ml_model'], model_input)
                
                st.info("AI Estimate (Alpha=0.10):")
                st.metric("Processing Time (p_i)", f"{pred_pi:.4f} Hours")
                
                # Store state to show the Save confirmation
                st.session_state['pending_save'] = {'record': full_record, 'pred': pred_pi}
            
            # Save Block (Appears only after prediction)
            if 'pending_save' in st.session_state:
                st.markdown("---")
                st.write("Does this prediction look accurate?")
                saved_state = st.session_state['pending_save']
                
                # Allow manual override before saving
                real_pi = st.number_input("Confirm Actual Time (Hours)", value=float(saved_state['pred']))
                
                if st.button("💾 Save to Database"):
                    record = saved_state['record']
                    record['p_i'] = real_pi
                    
                    if insert_new_item(record):
                        st.success(f"Part {record['part_id']} Saved Successfully!")
                        del st.session_state['pending_save']
                        st.session_state['ml_model'] = None # Invalidate model so it retrains on new data
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error("Error saving to database (Part ID might already exist).")

    # --- TAB 2: DATA TABLE ---
    with tab2:
        st.subheader(f"Historical Data ({len(df_items)} Items)")
        st.dataframe(df_items, use_container_width=True, height=600)
        
        with st.expander("Dangerous Zone: Delete Item"):
            del_id = st.number_input("Enter ID to Delete", min_value=1, step=1)
            if st.button("❌ Delete Item"):
                if delete_item(del_id):
                    st.success(f"Part {del_id} deleted.")
                    st.session_state['ml_model'] = None # Retrain needed
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Delete failed. ID not found.")