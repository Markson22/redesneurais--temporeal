from ultralytics import YOLO
import cv2

# Carregando os modelos com verificação
try:
    modelo1 = YOLO('yolo11n.pt')  # Alterado para yolov8n.pt que é o modelo padrão
    print("Modelo 1 carregado com sucesso")
except Exception as e:
    print(f"Erro ao carregar modelo1: {e}")

try:
    modelo2 = YOLO('suspeitos.pt')
    print("Modelo 2 carregado com sucesso")
except Exception as e:
    print(f"Erro ao carregar modelo2: {e}")

try:
    modelo3 = YOLO('armas2.pt')
    print("Modelo 3 carregado com sucesso")
except Exception as e:
    print(f"Erro ao carregar modelo3: {e}")
