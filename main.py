import os
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

# Define o diretório para processar as imagens
diretorio = '/content'

# Loop pelos arquivos no diretório
for arquivo in os.listdir(diretorio):
    # Cria o caminho completo do arquivo
    caminho_imagem = f"{diretorio}/{arquivo}"

    # Lê a imagem
    img = cv2.imread(caminho_imagem)

    if img is not None:  # Verifica se a imagem foi lida corretamente
        # Redimensiona a imagem
        img = cv2.resize(img, (640, 480))

        # Converte a imagem de BGR para RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Analisa a imagem
        resultado = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        # Obtém as emoções e suas pontuações
        emocoes = resultado[0]['emotion']
        x = list(emocoes.values())
        y = list(emocoes.keys())

        # Cria a figura e os eixos
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plota as emoções
        ax2.barh(y, x)
        ax2.tick_params(labelsize=10)

        # Plota a imagem
        ax1.imshow(img)
        ax1.axis('off')  # Esconde os eixos para a imagem

        plt.show()  # Exibe a figura
