import sys
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QGraphicsOpacityEffect,
)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer, QEventLoop
from PySide6.QtGui import QFontDatabase


class ImprovedGui(QWidget):
    def __init__(self):
        super().__init__()
        self.loop = None  # Initialize loop attribute
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
        self.original_font = QFontDatabase().font("Eurostile-Roman", "Regular", 18)
        self.label.setFont(self.original_font)

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

    def calculate_timing(self, text: str) -> dict:
        """Calculate timing based on text length and reading speed"""
        char_count = len(text)
        word_count = len(text.split())
        
        # Timing calculations
        typing_speed_ms = 50  # milliseconds per character
        typing_duration = char_count * typing_speed_ms
        
        # Reading time based on 4.5 words per second (user's reading speed)
        reading_speed_wps = 4.5
        reading_duration = (word_count / reading_speed_wps) * 1000  # convert to milliseconds
        
        # Ensure reading time is at least as long as typing time
        reading_duration = max(reading_duration, typing_duration)
        
        # Animation durations
        fade_in_duration = 1000  # 1 second fade in
        display_duration = 1000  # 1 second to keep text visible after typing
        fade_out_duration = 1000  # 1 second fade out
        
        # Calculate when to start fade out
        # Use the longer of typing time or reading time to ensure user has enough time
        effective_content_time = max(typing_duration, reading_duration)
        fade_out_start = fade_in_duration + effective_content_time + display_duration
        
        # Total duration
        total_duration = fade_out_start + fade_out_duration
        
        return {
            "typing_duration": typing_duration,
            "reading_duration": reading_duration,
            "effective_content_time": effective_content_time,
            "fade_in_duration": fade_in_duration,
            "display_duration": display_duration,
            "fade_out_duration": fade_out_duration,
            "fade_out_start": fade_out_start,
            "total_duration": total_duration,
            "char_count": char_count,
            "word_count": word_count
        }

    def display_message(self, message, audio_duration_ms=None):
        """Display message with improved timing logic
        
        Args:
            message: Text to display
            audio_duration_ms: Optional audio duration in milliseconds for future TTS integration
        """
        # Calculate timing
        timing = self.calculate_timing(message)
        
        # If audio duration is provided (for future TTS), use it instead of calculated times
        if audio_duration_ms:
            # Use the longer of audio duration or typing duration
            effective_content_time = max(audio_duration_ms, timing["typing_duration"])
            timing["effective_content_time"] = effective_content_time
            timing["fade_out_start"] = timing["fade_in_duration"] + effective_content_time + timing["display_duration"]
            timing["total_duration"] = timing["fade_out_start"] + timing["fade_out_duration"]
        
        # Debug output (essential info only)
        print(f"📊 GUI: {timing['char_count']} chars, {timing['word_count']} words, {timing['effective_content_time']:.0f}ms display")
        
        # setup the event loop to know when the display is done
        self.loop = QEventLoop()

        # Reset font to original size for new message
        self.label.setFont(self.original_font)

        # Clean up any existing timers
        if hasattr(self, 'typing_timer'):
            self.typing_timer.stop()
            self.typing_timer.deleteLater()
        if hasattr(self, 'fade_out_timer'):
            self.fade_out_timer.stop()
            self.fade_out_timer.deleteLater()

        # Update the text message with the new message
        self.label.setText("")
        
        self.typing_message = message
        self.typing_index = 0

        # Set up the typing animation timer
        self.typing_timer = QTimer(self)
        self.typing_timer.timeout.connect(self.type_character)
        self.typing_timer.start(50)  # 50ms per character

        # Set up the fade-in animation
        self.animation.setDuration(timing["fade_in_duration"])
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.start()

        # Set up a QTimer to call the fade-out animation at the right time
        self.fade_out_timer = QTimer(self)
        self.fade_out_timer.setSingleShot(True)
        self.fade_out_timer.timeout.connect(lambda: self.fade_out(timing["fade_out_duration"]))
        self.fade_out_timer.start(timing["fade_out_start"])

    def fade_out(self, duration=1000):
        """Start fade out animation"""
        self.animation.setDuration(duration)
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
        """Wait for all animations to complete"""
        if self.loop is not None:
            self.loop.exec()
        else:
            # No active animation loop, nothing to wait for
            pass

    def sleep(self, duration):  # duration in milliseconds
        """Non-blocking sleep for event loop"""
        self.sleep_loop = QEventLoop()
        QTimer.singleShot(duration, self.sleep_loop.quit)
        self.sleep_loop.exec()
