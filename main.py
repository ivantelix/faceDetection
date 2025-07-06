# main.py
import cv2
from detector.shape_detector import detectar_formas
from detector.person_detector import detectar_personas
from detector.face_identifier import identificar_rostros

# Cerrar todas las ventanas previas
cv2.destroyAllWindows()

# Inicializa la cámara
cap = cv2.VideoCapture(0)  # Cambia a 1 o 2 según la cámara

if not cap.isOpened():
    print("❌ No se pudo abrir la cámara.")
    exit()

print("✅ Cámara iniciada. Presiona 'q' para salir.")

# Crear una sola ventana
cv2.namedWindow("Detección en tiempo real", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detección en tiempo real", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No se pudo leer el frame.")
        break

    # Detectar figuras geométricas
    detectar_formas(frame)

    # Detectar personas (YOLO)
    personas_detectadas = detectar_personas(frame)

    # Identificar rostros (face_recognition)
    identificar_rostros(frame, personas_detectadas)

    # Mostrar resultados
    cv2.imshow("Detección en tiempo real", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()