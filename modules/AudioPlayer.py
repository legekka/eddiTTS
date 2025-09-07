import wave
import pyaudio
import threading
import time
from typing import Optional, Callable, Dict, Any
import logging


class AudioPlayer:
    """Audio player that uses the system's default audio device"""
    
    def __init__(self, config: Dict = None):
        """Initialize the audio player"""
        self.config = config or {}
        self.pyaudio_instance = pyaudio.PyAudio()
        self.logger = logging.getLogger(__name__)
        self.is_playing = False
        self.current_stream = None
        
        # Get default audio device
        self.default_device = self._get_default_audio_device()
        
    def _get_default_audio_device(self) -> Optional[Dict[str, Any]]:
        """Get the system's default audio output device"""
        try:
            # Get default output device
            device_info = self.pyaudio_instance.get_default_output_device_info()
            self.logger.info(f"Default audio device: {device_info['name']} (index: {device_info['index']})")
            return device_info
        except Exception as e:
            self.logger.error(f"Failed to get default audio device: {e}")
            return None
    
    def get_audio_info(self, audio_data: bytes) -> Dict[str, Any]:
        """Get audio file information from wave data"""
        try:
            import io
            audio_io = io.BytesIO(audio_data)
            with wave.open(audio_io, 'rb') as wf:
                return {
                    'channels': wf.getnchannels(),
                    'sample_width': wf.getsampwidth(),
                    'framerate': wf.getframerate(),
                    'frames': wf.getnframes(),
                    'duration': wf.getnframes() / wf.getframerate()
                }
        except Exception as e:
            self.logger.error(f"Failed to get audio info: {e}")
            return {}
    
    def play_audio_blocking(self, audio_data: bytes, on_start: Optional[Callable] = None, 
                           on_complete: Optional[Callable] = None) -> bool:
        """
        Play audio and block until playback is complete
        
        Args:
            audio_data: WAV audio data to play
            on_start: Callback function to call when playback starts
            on_complete: Callback function to call when playback completes
            
        Returns:
            True if playback was successful, False otherwise
        """
        if self.is_playing:
            self.logger.warning("Audio is already playing")
            return False
            
        try:
            import io
            audio_io = io.BytesIO(audio_data)
            
            with wave.open(audio_io, 'rb') as wf:
                # Audio file info
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                frames = wf.getnframes()
                duration = frames / framerate
                
                self.logger.info(f"Playing audio: {duration:.2f}s, {channels}ch, {framerate}Hz")
                
                # Open audio stream
                stream = self.pyaudio_instance.open(
                    format=self.pyaudio_instance.get_format_from_width(sample_width),
                    channels=channels,
                    rate=framerate,
                    output=True,
                    output_device_index=self.default_device['index'] if self.default_device else None
                )
                
                self.current_stream = stream
                self.is_playing = True
                
                # Call start callback
                if on_start:
                    try:
                        on_start()
                    except Exception as e:
                        self.logger.error(f"Error in on_start callback: {e}")
                
                # Play audio in chunks
                chunk_size = 1024
                data = wf.readframes(chunk_size)
                
                while data and self.is_playing:
                    stream.write(data)
                    data = wf.readframes(chunk_size)
                
                # Clean up
                stream.stop_stream()
                stream.close()
                self.current_stream = None
                self.is_playing = False
                
                # Call completion callback
                if on_complete:
                    try:
                        on_complete()
                    except Exception as e:
                        self.logger.error(f"Error in on_complete callback: {e}")
                
                self.logger.info("Audio playback completed")
                return True
                
        except Exception as e:
            self.logger.error(f"Audio playback failed: {e}")
            self.is_playing = False
            self.current_stream = None
            return False
    
    def play_audio_async(self, audio_data: bytes, on_start: Optional[Callable] = None,
                        on_complete: Optional[Callable] = None) -> threading.Thread:
        """
        Play audio in a separate thread (non-blocking)
        
        Args:
            audio_data: WAV audio data to play
            on_start: Callback function to call when playback starts
            on_complete: Callback function to call when playback completes
            
        Returns:
            Thread object for the playback
        """
        def play_thread():
            self.play_audio_blocking(audio_data, on_start, on_complete)
        
        thread = threading.Thread(target=play_thread, daemon=True)
        thread.start()
        return thread
    
    def stop_playback(self):
        """Stop current audio playback"""
        if self.is_playing:
            self.is_playing = False
            if self.current_stream:
                try:
                    self.current_stream.stop_stream()
                    self.current_stream.close()
                except:
                    pass
                self.current_stream = None
            self.logger.info("Audio playback stopped")
    
    def close(self):
        """Clean up audio resources"""
        self.stop_playback()
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    # Legacy methods for backward compatibility
    def play(self, audio, sr):
        """Legacy method - convert to new interface if needed"""
        self.logger.warning("Using legacy play method - consider updating to new interface")
    
    def record(self, sample_rate=44100, time=5):
        """Legacy recording method"""
        self.logger.warning("Recording not implemented in new AudioPlayer")
        return None


if __name__ == "__main__":
    # Test the audio player
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Audio Player...")
    
    try:
        with AudioPlayer() as player:
            print(f"Default device: {player.default_device}")
            
            # Test with a TTS-generated file if it exists
            try:
                with open("test_tts_output.wav", "rb") as f:
                    audio_data = f.read()
                
                print("Playing test audio...")
                success = player.play_audio_blocking(
                    audio_data,
                    on_start=lambda: print("🔊 Audio started!"),
                    on_complete=lambda: print("🔊 Audio completed!")
                )
                
                if success:
                    print("Audio test successful!")
                else:
                    print("Audio test failed!")
                    
            except FileNotFoundError:
                print("No test audio file found (test_tts_output.wav)")
                
    except Exception as e:
        print(f"Error testing audio player: {e}")
