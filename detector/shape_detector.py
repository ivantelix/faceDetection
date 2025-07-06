# detector/shape_detector.py
import cv2
import numpy as np

def detectar_formas(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 100)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 200 or area > 5000:
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        lados = len(approx)

        forma = None
        color = (0, 255, 255)  # Amarillo por defecto

        if lados == 3:
            forma = "Triángulo"
            color = (255, 0, 0)  # Azul
        elif lados == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:
                forma = "Cuadrado"
                color = (0, 255, 0)  # Verde
        elif lados > 5:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius > 10:
                forma = "Círculo"
                color = (0, 0, 255)  # Rojo

        if forma:
            cv2.drawContours(frame, [approx], -1, color, 2)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.putText(frame, forma, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            print(f"[FORMA] {forma} detectado en ({x},{y})")
