import cv2
import numpy as np

def bhe2pl(image):
    """
    Ecualización de Bi-histograma usando dos límites de meseta (BHE2PL).
    Mejora el contraste preservando el brillo medio y evitando saturación.
    """
    # 1. SP (Punto de separación global)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    total_pixels = image.size
    p = hist / total_pixels
    
    k = np.arange(256)
    SP = int(np.round(np.sum(p * k)))
    
    lMIN = int(np.min(image))
    lMAX = int(np.max(image))
    
    if lMIN >= SP:
        SP = lMIN + 1
    if SP >= lMAX:
        SP = lMAX - 1
        
    # 3. SPL, SPU
    hl_sum = np.sum(hist[lMIN:SP+1])
    hu_sum = np.sum(hist[SP+1:lMAX+1])
    
    SPL = np.sum(k[lMIN:SP+1] * hist[lMIN:SP+1]) / hl_sum if hl_sum > 0 else 0
    SPU = np.sum(k[SP+1:lMAX+1] * hist[SP+1:lMAX+1]) / hu_sum if hu_sum > 0 else 0
    
    # 4. GR y D
    GRL1 = (SP - SPL) / (SP - lMIN) if (SP - lMIN) != 0 else 0
    DL = (1 - GRL1) / 2 if GRL1 > 0.5 else GRL1 / 2
    GRL2 = GRL1 + DL
    
    GRU1 = (lMAX - SPU) / (lMAX - SP) if (lMAX - SP) != 0 else 0
    DU = (1 - GRU1) / 2 if GRU1 > 0.5 else GRU1 / 2
    GRU2 = GRU1 + DU
    
    # 5. Límites de Meseta (Plateau Limits)
    PkL = np.max(hist[lMIN:SP+1]) if hl_sum > 0 else 0
    PkU = np.max(hist[SP+1:lMAX+1]) if hu_sum > 0 else 0
    
    PLL1 = GRL1 * PkL
    PLL2 = GRL2 * PkL
    PLU1 = GRU1 * PkU
    PLU2 = GRU2 * PkU
    
    # 6. Modificación del histograma
    mod_hist = np.zeros_like(hist)
    for i in range(lMIN, SP+1):
        if hist[i] <= PLL2:
            mod_hist[i] = PLL1
        else:
            mod_hist[i] = PLL2
            
    for i in range(SP+1, lMAX+1):
        if hist[i] <= PLU2:
            mod_hist[i] = PLU1
        else:
            mod_hist[i] = PLU2
            
    # 7. Ecualización independiente
    p_mod_L = mod_hist[lMIN:SP+1] / (np.sum(mod_hist[lMIN:SP+1]) + 1e-10)
    c_mod_L = np.cumsum(p_mod_L)
    
    f = np.zeros(256, dtype=np.uint8)
    X0_L = lMIN
    XL1_L = SP
    for idx, val in enumerate(range(lMIN, SP+1)):
        f_val = X0_L + (XL1_L - X0_L) * (c_mod_L[idx] - 0.5 * p_mod_L[idx])
        f[val] = np.clip(np.round(f_val), 0, 255)
        
    p_mod_U = mod_hist[SP+1:lMAX+1] / (np.sum(mod_hist[SP+1:lMAX+1]) + 1e-10)
    c_mod_U = np.cumsum(p_mod_U)
    
    X0_U = SP + 1
    XL1_U = lMAX
    for idx, val in enumerate(range(SP+1, lMAX+1)):
        f_val = X0_U + (XL1_U - X0_U) * (c_mod_U[idx] - 0.5 * p_mod_U[idx])
        f[val] = np.clip(np.round(f_val), 0, 255)
        
    # Llenar rangos fuera de min/max
    for i in range(256):
        if i < lMIN:
            f[i] = X0_L
        elif i > lMAX:
            f[i] = XL1_U
            
    img_out = cv2.LUT(image, f)
    return img_out
