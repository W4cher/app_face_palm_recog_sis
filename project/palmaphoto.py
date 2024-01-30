import cv2
from kivy.uix.popup import Popup
from kivy.uix.label import Label

def show_success_popup():
    popup = Popup(title='Sucesso',
                  content=Label(text='Mão cadastrada com sucesso!'),
                  size_hint=(None, None), size=(400, 200))
    popup.open()

def capture_and_save_photo(name, idfunc, cargo):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Pressione qualquer tecla para capturar a foto', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            file_name_path = f'hands/{name}_{idfunc}_{cargo}.jpg'
            cv2.imwrite(file_name_path, frame)
            print("Foto capturada e salva!")
            cap.release()
            cv2.destroyAllWindows()
            show_success_popup()
            break