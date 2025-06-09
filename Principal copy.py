from ultralytics import YOLO
import cv2

# Carregando os modelos
modelo1 = YOLO('yolo11n.pt')
modelo2 = YOLO('suspeitos.pt')
modelo3 = YOLO('armas2.pt')

# Definindo as classes para filtrar (todas as classes)
classes_modelo1 = None   # todas as classes do modelo YOLO11n
classes_modelo2 = None   # todas as classes do modelo de armas
classes_modelo3 = None   # todas as classes do 2º modelo de armas
# Mostra as classes disponíveis em cada modelo
print("\n=== Classes disponíveis ===")
print("\nModelo 1 (Pessoas):")
for idx, nome in modelo1.names.items():
    print(f"[{idx}] {nome}")

print("\nModelo 2 (Armas):")
for idx, nome in modelo2.names.items():
    print(f"[{idx}] {nome}")

print("\nModelo 3 (Armas):")
for idx, nome in modelo3.names.items():
    print(f"[{idx}] {nome}")        


# Fazendo predições com filtros aplicados
video = 'homemarmado2.mp4'

# Predições do primeiro modelo (com filtro)
resultado1 = modelo1.predict(video, 
                           conf=0.25,
                           show=False,
                           classes=classes_modelo1)  # aplicando filtro

# Predições do segundo modelo (com filtro)
resultado2 = modelo2.predict(video,
                           conf=0.25,
                           show=True,
                           classes=classes_modelo2)  # aplicando filtro


# Predições do terceiro modelo (com filtro)
resultado3 = modelo3.predict(video,         
                           conf=0.25,
                           show=True,
                           classes=classes_modelo3)  # aplicando filtro 

# Processando os resultados com os filtros aplicados
for r1, r2 in zip(resultado1, resultado2):
    # Verificando detecções filtradas
    if len(r1.boxes) > 0:
        print("\nDetecções Modelo 1 (Pessoas):")
        for box in r1.boxes:
            classe = int(box.cls[0])
            confianca = float(box.conf[0])
            print(f"Classe: {modelo1.names[classe]}, Confiança: {confianca:.2f}")
    
    if len(r2.boxes) > 0:
        print("\nDetecções Modelo 2 (Armas):")
        for box in r2.boxes:
            classe = int(box.cls[0])
            confianca = float(box.conf[0])
            print(f"Classe: {modelo2.names[classe]}, Confiança: {confianca:.2f}")

    if len(r2.boxes) > 0:
        print("\nDetecções Modelo 3 (Armas):")
        for box in r2.boxes:
            classe = int(box.cls[0])
            confianca = float(box.conf[0])
            print(f"Classe: {modelo3.names[classe]}, Confiança: {confianca:.2f}")


# Inicializar o vídeo
cap = cv2.VideoCapture(video)  # alterado de 0 para o arquivo de vídeo

while True:
    # Ler frame do vídeo
    success, frame = cap.read()
    if not success:
        break

    # Fazer predições com os três modelos
    resultado1 = modelo1.predict(frame, conf=0.25, classes=classes_modelo1)
    resultado2 = modelo2.predict(frame, conf=0.25, classes=classes_modelo2)
    resultado3 = modelo3.predict(frame, conf=0.25, classes=classes_modelo3)

    # Plotar resultados no frame
    for r in resultado1:
        frame = r.plot()
    for r in resultado2:
        frame = r.plot()
    for r in resultado3:
        frame = r.plot()

    # Mostrar o frame
    cv2.imshow('Detecção em Vídeo', frame)

    # Pressione 'q' para sair ou aguarde 1ms entre frames
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()