from ultralytics import YOLO
import cv2

# Carregando os modelos
modelo1 = YOLO('yolo11n.pt')
modelo4 = YOLO('crime_detection-2.pt')

# Configuração dos modelos
modelos = [modelo1, modelo4]
cores = [(0,255,0), (0,0,255)]  # Verde para pessoas, Vermelho para armas
confiancas = [0.25, 0.25]
classes = [[0], None]  # Apenas classe 0 (pessoas) para modelo1, todas as classes para modelo4

# Inicializar o vídeo
video = 'homemarmado.mp4'
cap = cv2.VideoCapture(video)

# Obtém as propriedades do vídeo original
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Configura o salvamento do vídeo
nome_saida = 'video_detectado.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(nome_saida, fourcc, fps, (largura, altura))

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Copia o frame original para não sobrescrever as detecções
    frame_final = frame.copy()
    
    # Processa cada modelo e desenha no mesmo frame
    for i, modelo in enumerate(modelos):
        resultados = modelo.predict(frame, 
                                  conf=confiancas[i], 
                                  classes=classes[i],  # Aplica o filtro de classes
                                  show=False)
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
                cv2.rectangle(frame_final, (x1, y1), (x2, y2), cores[i], 1)
                
                # Adiciona texto com a classe e confiança
                if i == 0:
                    label = f"Pessoa: {conf:.2f}"
                else:
                    label = f"Arma: {conf:.2f}"
                    
                cv2.putText(frame_final, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cores[i], 2)
    
    # Adiciona a legenda
    h, w = frame_final.shape[:2]
    cv2.putText(frame_final, "Pessoas (Verde)", (10, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(frame_final, "Armas (Vermelho)", (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    
    # Salva o frame processado no vídeo
    # out.write(frame_final)
    
    # Mostra o frame com todas as detecções
    cv2.imshow('Detecções Combinadas', frame_final)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
cap.release()
out.release()  # Fecha o arquivo de vídeo
cv2.destroyAllWindows()

print(f"Vídeo salvo como: {nome_saida}")