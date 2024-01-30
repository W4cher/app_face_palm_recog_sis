import tkinter as tk
from tkinter import PhotoImage
from PIL import Image, ImageTk
# from facial_recognition import start_facial_recognition
from palma_recogB import start_palm_recognition
# from test import start_palm_recognition

def resize_image(path, size=(20, 20)):
    image = Image.open(path)
    resized_image = image.resize(size, Image.ANTIALIAS)
    return ImageTk.PhotoImage(resized_image)

def create_window():
    window = tk.Tk()
    window.title("Sistemas de Reconhecimento Biometrico")
    window.geometry("400x400")  

    title = tk.Label(window, text="Sistemas de Reconhecimento", font=("Arial", 20))
    title.pack(pady=20)
    
    # Carrega, redimensiona e exibe a imagem da palma da mão
    face_image = resize_image("img/face.png")
    face_label = tk.Label(window, image=face_image)
    face_label.pack(pady=10)

    palm_image = resize_image("img/palm.png")
    palm_label = tk.Label(window, image=palm_image)
    palm_label.pack(pady=10)

    # os botões
    # btn_facial = tk.Button(window, text="Reconhecimento Facial", command=start_facial_recognition)
    # btn_facial.pack(pady=10)

    btn_palm = tk.Button(window, text="Reconhecimento de Palma", command=start_palm_recognition)
    btn_palm.pack(pady=10)
    
    

    window.mainloop()

if __name__ == "__main__":
    create_window()
