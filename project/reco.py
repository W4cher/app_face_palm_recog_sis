import kivy
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.event import EventDispatcher
from kivy.properties import ObjectProperty

kivy.require('1.9.1')

Window.clearcolor = (0.16, 0.16, 0.16)
Window.size = (600, 620)

def load_training_data(data_path):
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    training_data, labels = [], []
    for i, file in enumerate(onlyfiles):
        image_path = join(data_path, file)
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        training_data.append(np.asarray(images, dtype=np.uint8))
        labels.append(i)
    return np.asarray(training_data), np.asarray(labels, dtype=np.int32), onlyfiles

def train_face_recognizer(training_data, labels):
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(training_data, labels)
    return model

def face_detector(img, classifier, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img, []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi

class KivyCV(Image, EventDispatcher):
    on_identification_success = ObjectProperty(None)

    def __init__(self, capture, fps, model, classifier, file_list, **kwargs):
        super(KivyCV, self).__init__(**kwargs)
        self.capture = capture
        self.model = model
        self.classifier = classifier
        self.file_list = file_list
        self.is_identified = False
        Clock.schedule_interval(self.update, 1.0 / fps)

    def stop_camera_and_change_screen(self):
        self.capture.release()
        self.manager.current = 'ValidationScreen'
        Clock.schedule_once(self.restart_camera, 10)

    def restart_camera(self, dt):
        self.capture.open(0)
        
    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return  

        image, face = face_detector(frame, self.classifier)

        # Tentar identificar o rosto
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = self.model.predict(face)

            confidence = min(100, int(100 * (1 - result[1] / 300)))
            if confidence > 85:
                identified_label = self.file_list[result[0]].split("_")[0]
                # display_string = f'{confidence}% Confidence it is {identified_label}'
                self.identify_success(image, identified_label, confidence)
                # self.identify_success(image, identified_label, confidence)
            else:
                self.identify_fail(image)

        except Exception as e:
            # Se não houver rosto ou ocorrer um erro
            cv2.putText(image, "Rosto desconhecido", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        self.display_image(image)

    def identify_success(self, image, identified_label, confidence):
        # cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
        cv2.putText(image, identified_label, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)
        if not self.is_identified:
            print('identificado')
            self.is_identified = True
            # self.stop_camera_and_change_screen()
            popup = IdentificationPopup(confidence=confidence, identified_label=identified_label)
            popup.open()

    def identify_fail(self, image, display_string):
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
        cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    def display_image(self, image):
        buf = cv2.flip(image, 0).tobytes()
        image_texture = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = image_texture


class FunctionScreen(Screen):
    def __init__(self, model, classifier, file_list, **kwargs):
        super(FunctionScreen, self).__init__(**kwargs)
        self.model = model
        self.classifier = classifier
        self.file_list = file_list
        self.layout = FloatLayout()
        self.add_widget(self.layout)

        self.setup_camera()
        self.create_ui_elements()

    def setup_camera(self):
        # Configuração da câmera
        self.capture = cv2.VideoCapture(0)
        self.kivy_cv = KivyCV(capture=self.capture, fps=30, model=self.model,
                              classifier=self.classifier, file_list=self.file_list)
        self.kivy_cv.size_hint = (1, 0.8)
        self.kivy_cv.pos_hint = {'x': 0, 'y': 0.2}
        self.layout.add_widget(self.kivy_cv)

    def create_ui_elements(self):
        # Botão para iniciar o reconhecimento facial
        start_button = Button(
            text='Iniciar Reconhecimento',
            size_hint=(0.4, 0.1),
            pos_hint={'center_x': 0.5, 'center_y': 0.1}
        )
        start_button.bind(on_press=self.start_recognition)
        self.layout.add_widget(start_button)


    def start_recognition(self, instance):
        print("Iniciando reconhecimento facial...")

        # Ativar ou reiniciar a captura da câmera
        if not self.capture.isOpened():
            self.capture.open(0)

        # Resetar flag de identificação
        self.kivy_cv.is_identified = False

        # Atualizar interface do usuário se necessário
        # Por exemplo, alterar o texto do botão ou desativá-lo durante o reconhecimento
        instance.text = 'Reconhecimento em Andamento...'
        # instance.disabled = True

        # Outras configurações iniciais para o reconhecimento podem ser feitas aqui
        # Por exemplo, inicializar variáveis, configurar parâmetros, etc.

        # Iniciar ou retomar o loop de atualização da imagem da câmera
        Clock.schedule_interval(self.kivy_cv.update, 1.0 / 30)  # 30 FPS

    def go_back(self, instance):
        self.manager.transition.direction = 'right'
        self.manager.current = 'ValidationScreen'  # Substitua pelo nome da tela anterior

    def on_leave(self):
        # Garantir que a câmera seja liberada quando a tela for deixada
        self.capture.release()


class ValidationScreen(Screen):
    def __init__(self, **kwargs):
        super(ValidationScreen, self).__init__(**kwargs)

        self.layout = FloatLayout()
        self.add_widget(self.layout)

        self.create_ui_elements()

    def create_ui_elements(self):
        # Título da tela
        title_label = Label(
            text='Registro de Rosto',
            size_hint=(0.6, 0.2),
            pos_hint={'top': 1, 'center_x': 0.5}
        )
        self.layout.add_widget(title_label)

        # Botão para iniciar o registro de rosto
        register_button = Button(
            text='Iniciar Registro',
            size_hint=(0.3, 0.1),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        register_button.bind(on_press=self.start_registration)
        self.layout.add_widget(register_button)

        # Botão para retornar à tela principal
        back_button = Button(
            text='Voltar',
            size_hint=(0.2, 0.1),
            pos_hint={'center_x': 0.5, 'y': 0.1}
        )
        back_button.bind(on_press=self.go_back)
        self.layout.add_widget(back_button)

    def start_registration(self, instance):
        print('Iniciando o registro de rosto...')
        # Aqui você pode adicionar a lógica para iniciar o registro de rosto
        # Por exemplo, mudar para uma tela que captura e salva as imagens dos rostos

    def go_back(self, instance):
        self.manager.transition.direction = 'right'
        self.manager.current = 'ValidationScreen'  # Substitua pelo nome da tela anterior

class IdentificationPopup(Popup):
    def __init__(self, confidence, identified_label, **kwargs):
        super(IdentificationPopup, self).__init__(**kwargs)
        self.title = 'Identificação Confirmada'
        self.size_hint = (None, None)
        self.size = (400, 300)

        # Conteúdo do Popup
        self.content = self.create_content(confidence, identified_label)

    def create_content(self, confidence, identified_label):
        layout = FloatLayout()

        # Mensagem de confiança
        confidence_label = Label(
            text=f'Confiança: {confidence}%',
            size_hint=(0.8, 0.2),
            pos_hint={'center_x': 0.5, 'center_y': 0.7}
        )
        layout.add_widget(confidence_label)

        # Mensagem de identificação
        identification_label = Label(
            text=f'Identificado como: {identified_label}',
            size_hint=(0.8, 0.2),
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        layout.add_widget(identification_label)

        # Botão para fechar o popup
        close_button = Button(
            text='Fechar',
            size_hint=(0.6, 0.2),
            pos_hint={'center_x': 0.5, 'center_y': 0.3}
        )
        close_button.bind(on_press=self.dismiss_popup)
        layout.add_widget(close_button)

        return layout

    def dismiss_popup(self, instance):
        self.dismiss()


class SISTEMA(App):
    def build(self):
        training_data, labels, file_list = load_training_data('faces/')
        model = train_face_recognizer(training_data, labels)
        face_classifier = cv2.CascadeClassifier('lib/haarcascade_frontalface_default.xml')

        sm = ScreenManager()
        sm.add_widget(FunctionScreen(name='FunctionScreen', model=model, classifier=face_classifier, file_list=file_list))
        sm.add_widget(ValidationScreen(name='ValidationScreen'))
        return sm

if __name__ == '__main__':
    SISTEMA().run()
