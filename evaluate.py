import os
import cv2
import numpy as np
import scipy.stats as stats
import pandas as pd
from glob import glob
from skimage.measure import shannon_entropy
from bhe2pl import bhe2pl

def calculate_metrics(img_orig, img_enh):
    # AMBE
    ambe = abs(np.mean(img_orig, dtype=np.float64) - np.mean(img_enh, dtype=np.float64))
    
    # PSNR (OpenCV)
    psnr = cv2.PSNR(img_orig, img_enh)
    
    # Contraste (Desviación Estándar)
    contrast = np.std(img_enh, dtype=np.float64)
    
    # Entropía (Shannon)
    entropy = shannon_entropy(img_enh)
    
    return ambe, psnr, contrast, entropy

def evaluate_dataset(image_dir="dataset"):
    image_paths = glob(os.path.join(image_dir, '*.*'))
    # Filtro de formatos
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not image_paths:
        print(f"No se encontraron imágenes en {image_dir}")
        return
    
    total_images = len(image_paths)
    results = []
    
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    for i, path in enumerate(image_paths, start=1):
        print(f"Evaluando imagen {i}/{total_images}")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # 1. Ecualización de Histograma (HE)
        he_img = cv2.equalizeHist(img)
        # 2. CLAHE
        clahe_img = clahe_obj.apply(img)
        # 3. BHE2PL
        bhe_img = bhe2pl(img)
        
        # Métricas HE
        ambe_he, psnr_he, c_he, e_he = calculate_metrics(img, he_img)
        # Métricas CLAHE
        ambe_clahe, psnr_clahe, c_clahe, e_clahe = calculate_metrics(img, clahe_img)
        # Métricas BHE2PL
        ambe_bhe, psnr_bhe, c_bhe, e_bhe = calculate_metrics(img, bhe_img)
        
        results.append({
            'filename': os.path.basename(path),
            'HE_AMBE': ambe_he, 'CLAHE_AMBE': ambe_clahe, 'BHE2PL_AMBE': ambe_bhe,
            'HE_PSNR': psnr_he, 'CLAHE_PSNR': psnr_clahe, 'BHE2PL_PSNR': psnr_bhe,
            'HE_Contrast': c_he, 'CLAHE_Contrast': c_clahe, 'BHE2PL_Contrast': c_bhe,
            'HE_Entropy': e_he, 'CLAHE_Entropy': e_clahe, 'BHE2PL_Entropy': e_bhe
        })
        
    df = pd.DataFrame(results)
    
    print("\n--- Promedios Globales ---")
    print(df.mean(numeric_only=True))
    
    print("\n--- Pruebas Estadísticas (p-valor) ---")
    print("T-test relacionando CLAHE vs BHE2PL (buscando si hay deferencias reales)")
    
    for metric in ['AMBE', 'PSNR', 'Contrast', 'Entropy']:
        col1 = f"CLAHE_{metric}"
        col2 = f"BHE2PL_{metric}"
        t_stat, p_val = stats.ttest_rel(df[col1], df[col2])
        sig = "Significativo" if p_val < 0.05 else "NO Significativo"
        print(f"{metric}: p-value = {p_val:.4e} ({sig})")
        
    return df

if __name__ == "__main__":
    evaluate_dataset()
