import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm

# --- CONFIGURATION ---
ALPHA_TO_REMOVE = 0.10  # Strict significance level (matches Minitab)

def train_best_subset_model(df):
    """
    Performs Best Subset Selection with strict P-value filtering.
    Only models where ALL variables have P-value < 0.10 are considered.
    Among those, the one with the best Adjusted R-squared is chosen.
    """
    # 1. Safety Checks
    if df.empty or len(df) < 10:
        return None, "Not enough data to train (need at least 10 items)."

    df_clean = df.copy()
    target = 'p_i'
    
    # 2. Define Candidate Features (The Dummy Vars are now explicit)
    candidates = [
        'w_i', 'h_i', 'l_i', 'paint_used', 
        'dbasic', 'dsmall', 'dmedium', 'dlarge'
    ]
    
    # Filter to ensure they exist
    available_features = [f for f in candidates if f in df_clean.columns]
    
    y = df_clean[target]
    
    best_adj_r2 = -np.inf
    best_model_info = None
    
    # 3. Exhaustive Search
    # Iterate through all combinations of features
    for k in range(1, len(available_features) + 1):
        for combo in itertools.combinations(available_features, k):
            features = list(combo)
            X = df_clean[features]
            X = sm.add_constant(X) # Statsmodels requires explicit intercept
            
            try:
                # Fit OLS Model
                model = sm.OLS(y, X).fit()
                
                # --- P-VALUE CHECK (The Minitab Logic) ---
                # Get p-values for all variables (exclude 'const' from strict check if desired, 
                # but usually we check predictors. We skip index 0 which is const)
                p_values = model.pvalues[1:] 
                
                # If ANY variable has P > 0.10, discard this model
                if (p_values > ALPHA_TO_REMOVE).any():
                    continue
                
                # If valid, check R-squared
                adj_r2 = model.rsquared_adj
                
                if adj_r2 > best_adj_r2:
                    best_adj_r2 = adj_r2
                    best_model_info = {
                        'model': model,
                        'features': features,
                        'adj_r2': adj_r2,
                        'p_values': model.pvalues
                    }
            except:
                continue
    
    if best_model_info is None:
        return None, f"No model found where all variables have P-value < {ALPHA_TO_REMOVE}"
        
    return best_model_info, None

def predict_processing_time(model_info, input_dict):
    """
    Predicts p_i using the best statsmodels object.
    """
    model = model_info['model']
    features = model_info['features']
    
    # Prepare Input
    df_in = pd.DataFrame([input_dict])
    X_in = df_in[features]
    X_in = sm.add_constant(X_in, has_constant='add') 
    
    # Statsmodels prediction requires the columns to match exactly
    # We must force the 'const' column to be present (value 1.0)
    X_in['const'] = 1.0
    
    # Reorder columns to match training data
    # The model params index gives the exact order: ['const', 'w_i', ...]
    train_cols = model.params.index.tolist()
    X_final = X_in[train_cols]
    
    return model.predict(X_final)[0]