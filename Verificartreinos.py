from ultralytics import YOLO

# Carrega o modelo YOLO
modelo = YOLO('yolov8.pt')

# Imprime as classes que o modelo pode detectar
print("Classes que o modelo pode detectar:")
print(modelo.names)

# Imprime informações detalhadas do modelo
print("\nInformações do modelo:")
print(modelo.info())