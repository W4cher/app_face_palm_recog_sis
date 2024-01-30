import cv2
import kivy

kivy.require('1.9.1') 

from kivy.lang import Builder
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from palmaphoto import capture_and_save_photo
from palma_recogB import start_palm_recognition
from kivy.uix.popup import Popup
from kivy.uix.label import Label


# COR DA JANELA E TAMANHO
Window.clearcolor = (0.5, 0.5, 0.5, 1)
Window.size = (980, 720)

# CAMERA NO KIVY CONFIGURAÇÃO
class KivyCV(Image):
    def __init__(self, capture, fps, **kwargs):
        Image.__init__(self, **kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    # CONFIGURAÇÃO PARA DETECTAR FACE
    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faceCascade = cv2.CascadeClassifier("lib/haarcascade_frontalface_default.xml")
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(20, 20))
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # TRANSFORMANDO UMA IMAGEM EM TEXTURA PARA COLOCAR A CAMERA
            buf = cv2.flip(frame, 0).tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # display image from the texture
            self.texture = image_texture


# FUNÇÃO DO SCREAN PARA MUDAR DE TELA
class SISTEMA(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(WelcomeScreen(name='welcomeScreen'))
        # sm.add_widget(FunctionScreen(name='functionScreen'))
        return sm

# PRIMEIRA TELA DO SUB-SCREEN
class WelcomeScreen(Screen):
    def __init__(self, **kwargs):
        super(WelcomeScreen, self).__init__(**kwargs)
        layout1 = FloatLayout()
        box = BoxLayout(orientation='horizontal', size_hint=(0.4, 0.2), padding=8, pos_hint={'top': 0.2, 'center_x': 0.5})
        tituloscreen1 = Label(text='CLIQUE EM FOTOS PARA CADASTRAR AS FACES PARA O RECONHECIMENTO',color =[1, 0, 0, 1], halign='center',
                              font_name = 'Roboto-Bold', valign='center', size_hint=(0.4, 0.2), pos_hint={'top': 0.3, 'center_x': 0.5})

        # TILULO DO PROGRAMA
        self.title1 = Label(text='SISTEMA DE CADASTRO')
        self.title1.font_size = '60sp'
        self.title1.color = [1, 25, 91, 1]
        self.title1.font_name = 'Roboto-Bold'
        self.title1.size_hint = (.99, .99)
        self.title1.pos_hint = {'x': .0, 'y': .40}

        # CONFIGURAÇÃO DO BOTÃO TIRAR FOTOS
        # FOTO = Button(text='FOTOS', on_press=self.tirarfoto)

        # CONFIGURAÇÃO DO BOTÃO CADASTRO
        # CADASTRAR = Button(text='CADASTRAR', on_press=self.cadastrar)
        CADASTRAR = Button(text='CADASTRAR ROSTO', on_press=self.fotofaces)
        # CONFIGURAÇÃO DO BOTÃO TIRAR FOTO
        self.BOTAO_FOTO = Button(text='CADASTRAR MÃO', on_press=self.tirar_foto)
        box.add_widget(self.BOTAO_FOTO)



        # CAMPO NOME: CAIXA E TEXTO INPUT
        self.caixatexto = (Label(text="NOME:"))
        self.caixatexto.size_hint = (.005, .07)
        self.caixatexto.font_size = 26
        self.caixatexto.pos_hint = {'x': .21, 'y': .72}
        self.username = TextInput(multiline=False)
        self.username.write_tab = False
        self.username.size_hint = (.5, .07)
        self.username.font_size = 26
        self.username.pos_hint = {'x': .30, 'y': .72}

        # CAMPO ID: CAIXA E TEXTO INPUT
        self.caixatextoidfunc = (Label(text="ID:"))
        self.caixatextoidfunc.size_hint = (.005, .07)
        self.caixatextoidfunc.font_size = 26
        self.caixatextoidfunc.pos_hint = {'x': .21, 'y': .60}
        self.idfunc = TextInput(multiline=False)
        self.idfunc.write_tab = False
        self.idfunc.size_hint = (.5, .07)
        self.idfunc.font_size = 26
        self.idfunc.pos_hint = {'x': .30, 'y': .60}

        # CAMPO CARGO: CAIXA E TEXTO INPUT
        self.caixatextocargo = (Label(text="CARGO:"))
        self.caixatextocargo.size_hint = (.005, .07)
        self.caixatextocargo.font_size = 26
        self.caixatextocargo.pos_hint = {'x': .21, 'y': .48}
        self.cargo = TextInput(multiline=False)
        self.cargo.write_tab = False
        self.cargo.size_hint = (.5, .07)
        self.cargo.font_size = 26
        self.cargo.pos_hint = {'x': .30, 'y': .48}

        # LAYOUT PARA MOSTRAR OS WIDGET NA TELA
        # WIDGETS BOXLAYOUT
        box.add_widget(CADASTRAR)
        # box.add_widget(FOTO)

        # WIDGETES FLOATLAYOUT
        layout1.add_widget(box)
        layout1.add_widget(self.title1)
        layout1.add_widget(self.username)
        layout1.add_widget(self.caixatexto)
        layout1.add_widget(self.idfunc)
        layout1.add_widget(self.caixatextoidfunc)
        layout1.add_widget(self.cargo)
        layout1.add_widget(self.caixatextocargo)
        layout1.add_widget(tituloscreen1)

        # CONFIGURAÇÃO LAYOUT1
        self.add_widget(layout1)


    # FUNÇÃO DO CLIQUE CADASTRAR
    def cadastrar(self, instance):
        name = self.username.text
        idfunc = self.idfunc.text
        cargo = self.cargo.text
        print("Name:", name, "\nCPF:", idfunc, "\nCargo:", cargo)

        print('CADASTRO EFETUADO COM SUCESSO')

    def tirar_foto(self, instance):
        capture_and_save_photo(self.username.text, self.idfunc.text, self.cargo.text)
        start_palm_recognition(self.username.text, self.idfunc.text, self.cargo.text)
        
    



    # FUNÇÃO DO CLIQUE PARA EXTRAIR AS FACES
    def fotofaces(self, *args):
        def show_success_popup():
            popup = Popup(title='Sucesso',
                        content=Label(text='10 Rostos cadastrados com sucesso!'),
                        size_hint=(None, None), size=(400, 200))
            popup.open()
            
        print('VOCE CLICOU NO BOTÃO TIRAR FOTOS')
        name = self.username.text
        idfunc = self.idfunc.text
        cargo = self.cargo.text
        

        # CODIGO PARA EXTRAIR AS IMAGENS
        def face_extractor(img):

            face_classifier = cv2.CascadeClassifier("lib/haarcascade_frontalface_default.xml")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            if faces is ():
                return None

            for (x, y, w, h) in faces:
                cropped_face = img[y:y + h, x:x + w]

            return cropped_face

        cap = cv2.VideoCapture(0)
        count = 0

        while True:
            ret, frame = cap.read()
            if face_extractor(frame) is not None:
                count += 1
                face = cv2.resize(face_extractor(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                # file_name_path = 'faces/user' + str(count) + '.jpg'
                file_name_path = 'faces/' + name + '_' + idfunc + '_' + cargo + '_' + str(count) + '.jpg'
                cv2.imwrite(file_name_path, face)

                cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', face)
            else:
                print("Face not Found")
                pass

            if cv2.waitKey(1) == 13 or count == 10:
                break
        show_success_popup()
        cap.release()
        cv2.destroyAllWindows()
        print('Colleting Samples Complete!!!')
            

if __name__ == '__main__':
    SISTEMA().run()
