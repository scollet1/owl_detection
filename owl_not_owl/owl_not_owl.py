from kivy.app import App
from kivy.lang import Builder
from kivy.config import Config
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition

class ScreenManagement(ScreenManager):
	pass

class HomeScreen(Screen):
	pass

class DetectionScreen(Screen):
	# pass
	def __init__(self, **kwargs):
		super(DetectionScreen, self).__init__(**kwargs)
		print (kwargs)
	def quepedo(self, qp):
		print (qp.text)


owls = Builder.load_file("owl_not_owl.kv")

class Owl_Not_Owl(App):
	def build(self):
		return owls

if __name__ == "__main__":
	Owl_Not_Owl().run()
