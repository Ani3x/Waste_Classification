import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.feature import local_binary_pattern


def extract_features(image_path):

    img = cv2.imread(image_path)
    
    if img is None: 
        return None
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # CECHY KOLORU PO KOLEI

    # Histogramy HSV (H - odcień, S - nasycenie, V - jasność)
    hist_h = cv2.calcHist([img_hsv], [0], None, [16], [0, 180])
    hist_s = cv2.calcHist([img_hsv], [1], None, [16], [0, 256])
    hist_v = cv2.calcHist([img_hsv], [2], None, [16], [0, 256])

    #Normalizacja
    hist_h = hist_h / (hist_h.sum() + 1e-6)
    hist_s = hist_s / (hist_s.sum() + 1e-6)
    hist_v = hist_v / (hist_v.sum() + 1e-6)

    # Odchylenie standardowe (czy kolor jest jednolity czy taki ala poszarpany)
    std_color = cv2.meanStdDev(img_hsv)[1].flatten()

    # CECHY KSZTAŁTU - po kolei zamieniamy na szary (musi być żeby treshold działał), czarno biały i szukamy krawędzi 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Local_binary_pattern
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    hist = hist.astype("float") / (hist.sum() + 1e-6)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        # Współczynnik zwartości (Compactness)
        compactness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        # Prostokąt otaczający (Aspect Ratio)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h

    else:
        compactness, aspect_ratio = 0, 0



    # Wektor wszystkich cech
    features = np.concatenate([
    hist_h.flatten(),
    hist_s.flatten(),
    hist_v.flatten(),
    std_color,
    hist,  # LBP
    [compactness, aspect_ratio]])

    return torch.tensor(features, dtype=torch.float32)

