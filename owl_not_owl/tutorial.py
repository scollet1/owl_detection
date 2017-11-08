# import kivy
# "1.10.1.dev0" kivy version
# kivy.require('1.10.1.dev0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition

# from kivy.uix.floatlayout import FloatLayout
# from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import Line
# from kivy.uix.gridlayout import GridLayout
# from kivy.uix.textinput import TextInput
#

class Painter(Widget):
	def on_touch_down(self, touch):
		with self.canvas:
			touch.ud["line"] = Line(points=(touch.x, touch.y))

	def on_touch_move(self, touch):
		touch.ud["line"].points += [touch.x, touch.y]

class MainScreen(Screen):
	pass

class AnotherScreen(Screen):
	pass

class ScreenManagement(ScreenManager):
	pass

presentation = Builder.load_file("main.kv")

# class Loginscreen(GridLayout):
# 	def __init__(self, **kwargs):
# 		super(Loginscreen, self).__init__(**kwargs)
# 		self.cols = 2
#
# 		self.add_widget(Label(text="Username:"))
# 		self.username = TextInput(multiline=False)
# 		self.add_widget(self.username)
#
# 		self.add_widget(Label(text="Password:"))
# 		self.password = TextInput(multiline=False, password=True)
# 		self.add_widget(self.password)

# class Widgets(Widget):
# 	pass

# class DrawInput(Widget):
# 	def on_touch_down(self, touch):
# 		print (touch)
# 		with self.canvas:
# 			touch.ud["line"] = Line(points=(touch.x, touch.y))
#
# 	def on_touch_move(self, touch):
# 		print (touch)
# 		touch.ud["line"].points += (touch.x, touch.y)
#
# 	def on_touch_up(self, touch):
# 		print ("released\n", touch)

class MainApp(App):
	def build(self):
		return presentation

if __name__ == "__main__":
	MainApp().run()
