import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.config import Config

from predict_spatter import predict_spatter_map

Config.set('graphics', 'resizable', False)
Window.size = (800, 600)

class SpatterPredictionForm(GridLayout):
    def __init__(self, **kwargs):
        super(SpatterPredictionForm, self).__init__(**kwargs)
        self.cols = 2

        self.add_widget(Label(text="Laser Power (W):"))
        self.laser_power = TextInput(multiline=False)
        self.add_widget(self.laser_power)

        self.add_widget(Label(text="Head Position (mm):"))
        self.head_position = TextInput(multiline=False)
        self.add_widget(self.head_position)

        self.add_widget(Label(text="Welding Speed (mm/sec):"))
        self.welding_speed = TextInput(multiline=False)
        self.add_widget(self.welding_speed)

        self.add_widget(Label(text="Work Thickness (mm):"))
        self.work_thickness = TextInput(multiline=False)
        self.add_widget(self.work_thickness)

        self.calculate_button = Button(text="Calculate Spatter Map")
        self.calculate_button.bind(on_press=self.calculate_spatter_map)
        self.add_widget(self.calculate_button)

        self.exit_button = Button(text="Exit")
        self.exit_button.bind(on_press=self.exit_app)
        self.add_widget(self.exit_button)

    def calculate_spatter_map(self, instance):
        params = {
            'laser_power': self.laser_power.text,
            'head_position': self.head_position.text,
            'welding_speed': self.welding_speed.text,
            'work_thickness': self.work_thickness.text
        }
        output_file_name = predict_spatter_map(params)
        popup = Popup(title="Spatter Map", content=Image(source=output_file_name), size_hint=(0.8, 0.8))
        popup.open()

    def exit_app(self, instance):
        App.get_running_app().stop()
        Window.close()

class MyApp(App):
    def build(self):
        return SpatterPredictionForm()

if __name__ == '__main__':
    MyApp().run()