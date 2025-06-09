from ultralytics import YOLO
import cv2

# Carregando os modelos
modelo1 = YOLO('yolo11n.pt')  # Alterado de yolo11n.pt para yolov8n.pt (modelo padrão do YOLOv8)
modelo2 = YOLO('suspeitos.pt')


# Definindo as classes para filtrar
classes_modelo1 = [0,1]   # Classe 0 = pessoa no YOLOv8
classes_modelo2 = None  # todas as classes do modelo de armas

# Mostra as classes disponíveis em cada modelo
print("\n=== Classes disponíveis ===")
print("\nModelo 1 (Pessoas):")
for idx, nome in modelo1.names.items():
    print(f"[{idx}] {nome}")

print("\nModelo 2 (Armas):")
for idx, nome in modelo2.names.items():
    print(f"[{idx}] {nome}")


# Inicializar o vídeo
video = 'homemarmado2.mp4'
cap = cv2.VideoCapture(video)

# Configuração dos modelos
modelos = [modelo1, modelo2]
cores = [(0,255,0), (0,0,255)]  # Verde para pessoas, Vermelho para suspeitos
confiancas = [0.25, 0.25]

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Copia o frame original para não sobrescrever as detecções
    frame_final = frame.copy()
    
    # Processa cada modelo e desenha no mesmo frame
    for i, modelo in enumerate(modelos):
        resultados = modelo.predict(frame, conf=confiancas[i], show=False)
        for r in resultados:
            boxes = r.boxes
            for box in boxes:
                # Obtém as coordenadas da box
                x1, y1, x2, y2 = box.xyxy[0]
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Converte coordenadas para inteiros
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Desenha a box com a cor correspondente ao modelo
                cv2.rectangle(frame_final, (x1, y1), (x2, y2), cores[i], 2)
                
                # Adiciona texto com a classe e confiança
                if i == 0:
                    label = f"Pessoa: {conf:.2f}"
                elif i == 1:
                    label = f"Suspeito: {conf:.2f}"
                else:
                    label = f"Arma: {conf:.2f}"
                    
                cv2.putText(frame_final, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cores[i], 2)
    
    # Adiciona a legenda
    h, w = frame_final.shape[:2]
    cv2.putText(frame_final, "Pessoas (Verde)", (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.putText(frame_final, "Suspeitos (Vermelho)", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    
    # Mostra o frame com todas as detecções
    cv2.imshow('Detecções Combinadas', frame_final)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()