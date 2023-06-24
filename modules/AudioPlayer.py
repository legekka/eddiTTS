import sounddevice as sd


class AudioPlayer(object):
    def __init__(self, config):
        super().__init__()
        #sd.default.device = config["sounddevice"]["device"]

    def play(self, audio, sr):
        sd.play(audio, sr)
        #sd.wait()

    def record(self, sample_rate=44100, time=5):
        print("Recording...")
        audio = sd.rec(
            sample_rate * time, samplerate=sample_rate, channels=1, blocking=True
        )
        sd.wait()
        return audio
