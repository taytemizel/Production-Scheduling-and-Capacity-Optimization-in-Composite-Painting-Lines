import pandas as pd
import os
import random
import math

VERI_DOSYASI = 'veri.xlsx'
SURE_KOLONU = 'İŞLEM SÜRESİ (ZIMPARA+BOYA)'
PARCA_KOLONU = 'partno'
ZIMPARA_AGIRLIGI = 0.40
BOYA_AGIRLIGI = 0.60
TEST_MODU = True
TEST_ADEDI = 15

def load_and_prepare_data(filepath):
    print(f"Veri seti yükleniyor... ({filepath})")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HATA: '{filepath}' dosyası bulunamadı. Dosyanın klasörde olduğundan emin olun.")
    
    try:
        # Doğrudan Excel dosyasını okuyoruz
        df = pd.read_excel(filepath)
    except Exception as e:
        raise RuntimeError(f"Dosya okunurken hata oluştu (Terminale 'pip install openpyxl' yazdığınızdan emin olun): {e}")
        
    if SURE_KOLONU not in df.columns or PARCA_KOLONU not in df.columns:
        raise ValueError("HATA: Beklenen kolonlar bulunamadı. Lütfen Excel'deki başlıkları kontrol edin.")
        
    df = df.dropna(subset=[SURE_KOLONU])

    df['Zimpara_Suresi'] = df[SURE_KOLONU] * ZIMPARA_AGIRLIGI
    df['Boya_Suresi'] = df[SURE_KOLONU] * BOYA_AGIRLIGI
    
    if TEST_MODU:
        df = df.head(TEST_ADEDI).copy()
    job_list = df[PARCA_KOLONU].tolist()
    return df, job_list
def calculate_makespan(sequence, df, num_m1=2, num_m2=2):
    m1_available = [0] * num_m1
    m2_available = [0] * num_m2
    
    for job in sequence:
        p1 = df.loc[df['partno'] == job, 'Zimpara_Suresi'].values[0]
        p2 = df.loc[df['partno'] == job, 'Boya_Suresi'].values[0]
        
        earliest_m1_idx = m1_available.index(min(m1_available))
        start_m1 = m1_available[earliest_m1_idx]
        end_m1 = start_m1 + p1
        m1_available[earliest_m1_idx] = end_m1
        
        earliest_m2_idx = m2_available.index(min(m2_available))
        start_m2 = max(end_m1, m2_available[earliest_m2_idx])
        end_m2 = start_m2 + p2
        m2_available[earliest_m2_idx] = end_m2
        
    return max(m2_available)

def simulated_annealing(initial_sequence, df, num_m1=2, num_m2=2):
    current_sequence = initial_sequence.copy()
    current_makespan = calculate_makespan(current_sequence, df, num_m1, num_m2)
    
    best_sequence = current_sequence.copy()
    best_makespan = current_makespan
    
    T = 1000.0  
    T_min = 0.1 
    alpha = 0.95 
    
    while T > T_min:
        for i in range(50): 
            neighbor = current_sequence.copy()
            idx1, idx2 = random.sample(range(len(neighbor)), 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            
            neighbor_makespan = calculate_makespan(neighbor, df, num_m1, num_m2)
            delta = neighbor_makespan - current_makespan
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_sequence = neighbor.copy()
                current_makespan = neighbor_makespan
                
                if current_makespan < best_makespan:
                    best_sequence = current_sequence.copy()
                    best_makespan = current_makespan
                    
        T = T * alpha 
        
    return best_sequence, best_makespan

