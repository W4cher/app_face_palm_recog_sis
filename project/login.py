import os
import subprocess
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.floatlayout import FloatLayout

# 1. Importações Necessárias
# ...

# 2. Funções de Suporte

def obter_usuarios_da_pasta_faces():
    usuarios = []
    pasta_faces = 'faces/'
    for arquivo in os.listdir(pasta_faces):
        if arquivo.endswith('.jpg'):
            partes = arquivo.split('_')
            if len(partes) >= 3:
                nome = partes[0]
                id = partes[1]
                usuarios.append((nome, id))
    return usuarios

def verificar_login(nome, id):
    usuarios = obter_usuarios_da_pasta_faces()
    return any(usuario_nome == nome and usuario_id == id for usuario_nome, usuario_id in usuarios)

def executar_script_reconhecimento_rosto():
    subprocess.call(["python", "project/reco.py"])

def executar_script_reconhecimento_mao():
    subprocess.call(["python", "project/palma_recogB.py"])

# 3. Telas (Screens) do Kivy

class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super(LoginScreen, self).__init__(**kwargs)
        layout = FloatLayout()

        # Campos de texto e botões
        self.nome_input = TextInput(multiline=False, size_hint=(.8, .1), pos_hint={'x': .1, 'y': .6})
        self.id_input = TextInput(multiline=False, size_hint=(.8, .1), pos_hint={'x': .1, 'y': .4})
        botao_login = Button(text='Entrar', size_hint=(.8, .1), pos_hint={'x': .1, 'y': .2})
        botao_login.bind(on_press=self.on_login_pressed)

        layout.add_widget(self.nome_input)
        layout.add_widget(self.id_input)
        layout.add_widget(botao_login)

        self.add_widget(layout)

    def on_login_pressed(self, instance):
        nome = self.nome_input.text
        id = self.id_input.text
        if verificar_login(nome, id):
            self.manager.current = 'menu_principal'
        else:
            # Mostrar mensagem de erro
            print("Falha no Login")

class MenuPrincipalScreen(Screen):
    def __init__(self, **kwargs):
        super(MenuPrincipalScreen, self).__init__(**kwargs)
        layout = FloatLayout()

        # Botões
        botao_mao = Button(text='Reconhecimento por Mão', size_hint=(.8, .1), pos_hint={'x': .1, 'y': .6})
        botao_rosto = Button(text='Reconhecimento por Rosto', size_hint=(.8, .1), pos_hint={'x': .1, 'y': .4})
        botao_mao.bind(on_press=lambda x: executar_script_reconhecimento_mao())
        botao_rosto.bind(on_press=lambda x: executar_script_reconhecimento_rosto())

        layout.add_widget(botao_mao)
        layout.add_widget(botao_rosto)

        self.add_widget(layout)

# 4. Aplicativo Principal

class ReconhecimentoApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(MenuPrincipalScreen(name='menu_principal'))
        return sm

if __name__ == '__main__':
    ReconhecimentoApp().run()
