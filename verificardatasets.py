from ultralytics import YOLO

# Carregando os modelos
modelo1 = YOLO('yolo11n.pt')
modelo2 = YOLO('suspeitos.pt')
modelo3 = YOLO('runs\detect\weapon_detector_cpu\weights\best.pt')

# Definindo classes para filtrar em cada modelo
classes_modelo1 = [0]   # pessoa no modelo YOLO11n -> principal é 0 que é pessoas 2- bike, 3-carro, 4-moto, 5-ônibus, 6-truck, 7-avião, 8-navio, 9-caminhão

classes_modelo2 = ['0','1','2','3','4','5','6']  # seu modelo treinado (armas) -> 1- knife, 2-gun, 3-weapon, 4-gun2, 5-gun3, 6-gun4, 7-gun5

classes_modelo3 = [0]  # 2º modelo treinado (armas)

# Visualização das classes selecionadas
print("\n=== Classes Selecionadas ===")
print("\nModelo 1 (Yolo11n):")
for classe in classes_modelo1:
    print(f"- [{classe}] {modelo1.names[classe]}")

print("\nModelo 2 (suspeitos):")
for classe in classes_modelo2:
    print(f"- [{classe}] {modelo2.names[classe]}")      

print("\nModelo 3 (armas):")
for classe in classes_modelo3:
    print(f"- [{classe}] {modelo3.names[classe]}")              

# Fazendo predições com filtros
video = 'exemplo4arma.mp4'

# Predições do modelo 1 (armas)
resultado1 = modelo1.predict(video,
                           conf=0.25,
                           show=False,
                           classes=classes_modelo1)

# Predições do modelo 2 (pessoas)
resultado2 = modelo2.predict(video,
                           conf=0.25,
                           show=True,
                           classes=classes_modelo2)


# Predições do modelo 3 (armas)
resultado3 = modelo3.predict(video, 
                           conf=0.25,
                           show=True,
                           classes=classes_modelo3)

# Para adicionar mais classes do modelo2, modifique a lista classes_modelo2
# Exemplo para incluir pessoas e carros:
# classes_modelo2 = [0, 2]  # 0=pessoa, 2=carro