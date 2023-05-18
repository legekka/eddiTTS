import sys
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QGraphicsOpacityEffect,
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer, QEventLoop
from PySide6.QtGui import QFontDatabase


class Gui(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("AI Voice Assistant for Elite Dangerous")
        self.resize(450, 200)

        # remove window frame
        self.setWindowFlags(Qt.FramelessWindowHint)

        # position window in the top Right corner
        self.move(1920 - 450, 0)

        # set the window to be black
        self.setStyleSheet("background-color: black;")

        background = QLabel("", self)
        background.setStyleSheet(
            "background-image: url(res/background.png); background-repeat: no-repeat;"
        )
        background.setFixedWidth(450)
        background.setFixedHeight(200)
        background.move(0, 0)

        self.label = QLabel("Spoken text goes here", self)
        # align text to the bottom left
        self.label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        
        QFontDatabase().addApplicationFont("res/eurstl24.ttf")
        self.label.setFont(QFontDatabase().font("Eurostile-Roman", "Regular", 18))

        self.label.setStyleSheet(
            "color: #0a8bd6; padding: 4px; background-color: transparent;"
        )
       
        self.label.setFixedWidth(450)
        self.label.setFixedHeight(170)
        self.label.setWordWrap(True)
        # move label to the center of the window
        self.label.move(0, 30)

        self.opacity_effect = QGraphicsOpacityEffect()
        self.label.setGraphicsEffect(self.opacity_effect)

        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)

        # second animation for text for a Typing Effect
        self.animation2 = QPropertyAnimation(self.label, b"pos")
        self.animation2.setEasingCurve(QEasingCurve.InOutQuad)

    def display_message(self, message, duration):
        # setup the event loop to know when the update_message function is done
        self.loop = QEventLoop()

        # Update the text message with the new message
        self.label.setText("")
        
        self.typing_message = message
        self.typing_index = 0

        # Set up the typing animation timer
        self.typing_timer = QTimer(self)
        self.typing_timer.timeout.connect(self.type_character)
        self.typing_timer.start(50)  # Adjust the typing speed by changing this value

        # Set up the fade-in animation
        self.animation.setDuration(1000)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.start()

        # Set up a QTimer to call the fade-out animation after the fade-in animation is done
        self.fade_out_timer = QTimer(self)
        self.fade_out_timer.setSingleShot(True)
        self.fade_out_timer.timeout.connect(self.fade_out)
        self.fade_out_timer.start(duration - 1000)

    def fade_out(self):
        # Set up the fade-out animation
        self.animation.setDuration(1000)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.setEasingCurve(QEasingCurve.InQuad)
        self.animation.start()

        # wait for the fade-out animation to finish
        self.animation.finished.connect(self.loop.quit)

    # Typing animation effect
    def type_character(self):
        if self.typing_index < len(self.typing_message):
            current_text = self.label.text()
            current_text += self.typing_message[self.typing_index]
            self.label.setText(current_text)
            self.typing_index += 1

            # check if the text is too long to fit in the label
            if self.label.sizeHint().height() > self.label.height():
                # if it is, decrease the font size by 0.5
                font = self.label.font()
                font.setPointSize(font.pointSize() - 0.5)
                self.label.setFont(font)
        else:
            self.typing_timer.stop()

    def wait(self):
        self.loop.exec()

    def sleep(self, duration):  # duration in milliseconds
        self.sleep_loop = QEventLoop()
        QTimer.singleShot(duration, self.sleep_loop.quit)
        self.sleep_loop.exec()