import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import messagebox
import os

def get_user_info_from_files(folder_path):
    users = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):  # Verifica se é um arquivo de imagem
            parts = filename.split('_')
            if len(parts) >= 2:
                name = parts[0]
                user_id = parts[1]
                users.append((name, user_id, filename))
    return users

def start_palm_recognition():
    users = get_user_info_from_files('hands')
    if not users:
        print("Nenhum usuário encontrado na pasta 'hands'.")
        return

    # Exemplo: seleciona o primeiro usuário da lista
    user_name, user_id, filename = users[0]

    def display_match_success():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Verificação", f"Verificado com sucesso, obrigado {user_name}!")
        root.destroy()
        cap.release()
        cv2.destroyAllWindows()

    def load_reference_hand_image(path):
        ref_image = cv2.imread(path)
        if ref_image is None:
            print(f"Não foi possível carregar a imagem: {path}")
            return None
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        results = hands.process(ref_image)
        return results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

    def compare_hands(hand1, hand2, threshold=0.05, match_percentage=0.35):
        if hand1 is None or hand2 is None:
            return False
        matched_points = sum(np.sqrt((lm1.x - lm2.x) ** 2 + (lm1.y - lm2.y) ** 2) < threshold for lm1, lm2 in zip(hand1.landmark, hand2.landmark))
        return (matched_points / len(hand1.landmark)) >= match_percentage

    cap = cv2.VideoCapture(0)
    hands = mp.solutions.hands.Hands(max_num_hands=1)
    draw = mp.solutions.drawing_utils

    reference_hand = load_reference_hand_image(f'hands/{filename}')
    if reference_hand is None:
        print(f"Não foi possível carregar a imagem de referência para {user_name}.")
        return

    while True:
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                is_match = compare_hands(handLms, reference_hand, threshold=0.1, match_percentage=0.35)
                if is_match:
                    display_match_success()
                    return

                color = (0, 255, 0) if is_match else (0, 0, 255)
                draw.draw_landmarks(image, handLms, mp.solutions.hands.HAND_CONNECTIONS, draw.DrawingSpec(color=color, thickness=2, circle_radius=2))

        cv2.imshow("Hand", image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

start_palm_recognition()
