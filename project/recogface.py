import kivy
from kivy.event import EventDispatcher
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.clock import Clock


kivy.require('1.9.1')

from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# COR DA JANELA E TAMANHO
Window.clearcolor = (0, 0.1, 0, 1)
Window.size = (800, 720)

# DIRETORIO DAS IMAGENS
data_path = '../faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# TREINAMENTO FACES
Training_Data, Labels = [], []
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)
model = cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_Data), np.asarray(Labels))
print("TREINAMENTO EFETUADO")

face_classifier = cv2.CascadeClassifier('../lib/haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

class KivyCV(Image, EventDispatcher):
    on_identification_success = ObjectProperty(None)

    def __init__(self, capture, fps, **kwargs):
        Image.__init__(self, **kwargs)
        self.capture = capture
        self.is_identified = False  # Flag to track if identification has already occurred
        Clock.schedule_interval(self.update, 1.0 / fps)

    def stop_camera_and_change_screen(self):
        self.capture.release()  # Release the camera
        self.manager.current = 'ValidationScreen'  # Change to the ValidationScreen
        Clock.schedule_once(self.restart_camera, 10)  # Restart the camera after 10 seconds

    def restart_camera(self, dt):
        self.capture.open(0)  # Reopen the camera

    def update(self, dt):
        ret, frame = self.capture.read()
        image, face = face_detector(frame)
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)
            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
                if confidence > 75:
                    identified_label = onlyfiles[result[0]]
                    identified_label_parts = identified_label.split("_")
                    identified_label = " ".join(identified_label_parts[:-1])
                    display_string = str(confidence) + '% Confidence it is ' + identified_label
                    cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
                    cv2.putText(image, identified_label, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    # Dispare o evento de identificação bem-sucedida
                    self.is_identified = True  # Set the identification flag to True
                    self.stop_camera_and_change_screen()
                    popup = IdentificationPopup(confidence=confidence, identified_label=identified_label)
                    popup.open()
                else:
                    display_string = str(confidence) + '% Confidence it is user'
                    cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
                    cv2.putText(image, "BLOQUEADO", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        except:
            cv2.putText(image, "NAO CADASTRADO", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            pass

        def validado(self, instance):
            print('VOCE FOI PARA TELA 1')
            self.manager.current = 'ValidationScreen'

        buf = cv2.flip(frame, 0).tobytes()
        image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = image_texture
# FUNÇÃO DO SCREAN PARA MUDAR DE TELA
class SISTEMA(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(ValidationScreen(name='ValidationScreen'))
        sm.add_widget(FunctionScreen(name='functionScreen'))
        return sm


# PRIMEIRA TELA DO SUB-SCREEN
class FunctionScreen(Screen):
    def __init__(self, **kwargs):
        super(FunctionScreen, self).__init__(**kwargs)
        layout2 = FloatLayout()
        tituloscreen2 = Label(text='RECONHECIMENTO'
                                   '\n POSICIONE SEU ROSTO NO SENSOR',
                              halign='center', valign='center', size_hint=(0.4, 0.2),
                              font_size=40, font_name='Roboto-Bold', underline='True', outline_width=1,
                              outline_color=[1, 1, 1],
                              color=[1, 0, 0, 0.5], pos_hint={'top': 1, 'center_x': 0.5})

        # CONFIGURAÇÃO DO FLOATLAYOUT
        self.add_widget(layout2)

        # CONFIGURAÇÃO DA CAMERA
        self.capture = cv2.VideoCapture(0)
        self.my_camera = KivyCV(capture=self.capture, fps=60)
        # self.my_camera = KivyCV(capture=self.capture, fps=60, screen_manager=self.manager)
        self.my_camera.size_hint = (1, 1)
        self.my_camera.pos_hint = {'x': 0, 'y': .0}
  
        # LAYOUT PARA MOSTRAR OS WIDGET NA TELA
        layout2.add_widget(tituloscreen2)
        layout2.add_widget(self.my_camera)

# TERCEIRA TELA DO SCREEN
class ValidationScreen(Screen):
        def __init__(self, **kwargs):
            super(ValidationScreen, self).__init__(**kwargs)
            layout1 = FloatLayout()
            tituloscreen1 = Label(text='CLIQUE NO BOTÃO FOTO PARA REGISTRAR AS FACES NO BANDO DE DADOS'
                                       '\nATENÇÃO APÓS TIRAR AS 30 FOTOS CLIQUE EM VOLTAR PARA CONFIRMAR O CADASTRO',
                                  halign='center', valign='center', size_hint=(0.4, 0.2),
                                  color=[1, 0, 0, 1], pos_hint={'top': 1, 'center_x': 0.5})

            # CONFIGURAÇÃO DO FLOATLAYOUT
            self.add_widget(layout1)

            # CONFIGURAÇÕES DO BOTÃO VOLTAR
            self.botaoClick1 = Button(text='VOLTAR')
            self.botaoClick1.size_hint = (.2, .1)
            self.botaoClick1.pos_hint = {'x': .55, 'y': .50}
            # CONFIGURAÇÕES DO BOTÃO VOLTAR
            self.botaoClick2 = Button(text='Verficar')
            self.botaoClick2.size_hint = (.2, .1)
            self.botaoClick2.pos_hint = {'x': .95, 'y': .90}




            # CONFIGURAÇÕES DO BOTÃO TIRAR FOTOS

            # FUNÇÃO DE CLICK DO BOTÃO VOLTAR
            self.botaoClick1.bind(on_press=self.voltar)
            self.botaoClick2.bind(on_press=self.validar)

            # LAYOUT PARA MOSTRAR OS WIDGET NA TELA
            layout1.add_widget(tituloscreen1)
            layout1.add_widget(self.botaoClick1)
            layout1.add_widget(self.botaoClick2)

            # RETORNA PARA O WIDGET

        def validar(self, instance):
            print('VOCE FOI PARA TELA 2')
            self.manager.current = 'functionScreen'

        # FUNÇÃO DO CLIQUE VOLTAR
        def voltar(self, *args):
            print('VOCE CLICOU NO BOTÃO VOLTAR')
            self.manager.current = 'welcomeScreen'

class IdentificationPopup(Popup):
    def __init__(self, confidence, identified_label, **kwargs):
        super(IdentificationPopup, self).__init__(**kwargs)
        self.title = 'Identificação bem-sucedida'
        self.size_hint = (None, None)
        self.size = (400, 200)
        content = Label(text=f'Confiança: {confidence}%\nIdentificado como: {identified_label}')
        self.content = content


# FIM DO SISTEMA
if __name__ == '__main__':
    SISTEMA().run()
