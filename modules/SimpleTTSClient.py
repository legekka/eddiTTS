import requests
import time
import base64
import io
import wave
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple


class SimpleTTSClient:
    """Simple TTS client for sync-only TTS API"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SimpleTTS client
        
        Args:
            config: Configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        
        # TTS configuration
        tts_config = config.get("tts", {})
        self.base_url = tts_config.get("base_url", "http://172.16.240.5:4111")
        self.sid1 = tts_config.get("sid1", "22")
        self.sid2 = tts_config.get("sid2", "ganyu")
        self.timeout = tts_config.get("timeout", 30)
        
        # Ensure base_url doesn't have trailing slash
        self.base_url = self.base_url.rstrip('/')
        self.api_url = f"{self.base_url}/tts"
        
        self.logger.info(f"SimpleTTS Client initialized: {self.api_url}")
        self.logger.info(f"TTS settings: sid1={self.sid1}, sid2={self.sid2}")

    def generate_speech(self, text: str) -> bytes:
        """Generate speech from text and return WAV audio data
        
        Args:
            text: Text to convert to speech
            
        Returns:
            WAV audio data as bytes
            
        Raises:
            Exception: If TTS generation fails
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            # Prepare form data
            form_data = {
                "text": text,
                "sid1": self.sid1,
                "sid2": self.sid2
            }
            
            self.logger.debug(f"Generating TTS for: {text[:50]}...")
            start_time = time.time()
            
            # Make POST request to TTS API
            response = requests.post(
                self.api_url,
                data=form_data,
                timeout=self.timeout
            )
            
            generation_time = time.time() - start_time
            
            # Check response status
            response.raise_for_status()
            
            # Parse JSON response
            response_data = response.json()
            
            if "audio" not in response_data:
                raise Exception("No audio data in TTS response")
            
            # Decode base64 audio data
            audio_base64 = response_data["audio"]
            audio_bytes = base64.b64decode(audio_base64.encode("utf-8"))
            
            # Convert to WAV format (assuming the API returns raw audio)
            wav_data = self._convert_to_wav(audio_bytes)
            
            audio_size_kb = len(wav_data) / 1024
            self.logger.debug(f"TTS completed in {generation_time:.2f}s - {audio_size_kb:.1f}KB")
            
            return wav_data
            
        except requests.exceptions.Timeout:
            raise Exception(f"TTS request timed out after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            raise Exception(f"TTS API request failed: {e}")
        except Exception as e:
            raise Exception(f"TTS generation failed: {e}")

    def _convert_to_wav(self, audio_bytes: bytes) -> bytes:
        """Convert raw audio bytes to WAV format
        
        Args:
            audio_bytes: Raw audio data
            
        Returns:
            WAV formatted audio data
        """
        try:
            # Try to read as various audio formats
            try:
                import soundfile as sf
                soundfile_available = True
            except ImportError:
                soundfile_available = False
            
            if soundfile_available:
                # Use soundfile if available
                audio_io = io.BytesIO(audio_bytes)
                audio_io.seek(0)
                
                # Read audio data
                audio_data, sample_rate = sf.read(audio_io, dtype='float32')
                
                # Convert to 16-bit PCM
                if audio_data.dtype != np.int16:
                    # Normalize to [-1, 1] if needed
                    if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                        audio_data = audio_data / np.max(np.abs(audio_data))
                    
                    # Convert to 16-bit PCM
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                else:
                    audio_int16 = audio_data
                
                # Create WAV file in memory
                wav_io = io.BytesIO()
                with wave.open(wav_io, 'wb') as wav_file:
                    # Determine number of channels
                    if len(audio_int16.shape) == 1:
                        num_channels = 1
                        frames_data = audio_int16
                    else:
                        num_channels = audio_int16.shape[1]
                        frames_data = audio_int16
                    
                    wav_file.setnchannels(num_channels)
                    wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                    wav_file.setframerate(int(sample_rate))
                    wav_file.writeframes(frames_data.tobytes())
                
                return wav_io.getvalue()
            else:
                # Fallback: assume it's already WAV format or raw PCM
                self.logger.warning("soundfile not available, assuming audio is already in WAV format")
                return audio_bytes
            
        except Exception as e:
            self.logger.error(f"Failed to convert audio to WAV: {e}")
            # Fallback: assume it's already WAV format
            return audio_bytes

    def save_audio_to_file(self, audio_data: bytes, filename: str):
        """Save audio data to a file
        
        Args:
            audio_data: WAV audio data
            filename: Output filename
        """
        try:
            with open(filename, 'wb') as f:
                f.write(audio_data)
            self.logger.debug(f"Audio saved to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save audio to {filename}: {e}")

    def get_audio_info(self, audio_data: bytes) -> Dict[str, Any]:
        """Get audio file information from WAV data
        
        Args:
            audio_data: WAV audio data
            
        Returns:
            Dictionary with audio information
        """
        try:
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

    def test_connection(self) -> bool:
        """Test connection to TTS API
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test with a simple text
            test_text = "Test"
            audio_data = self.generate_speech(test_text)
            
            # Check if we got valid audio data
            if len(audio_data) > 0:
                self.logger.info("TTS API connection test successful")
                return True
            else:
                self.logger.error("TTS API returned empty audio data")
                return False
                
        except Exception as e:
            self.logger.error(f"TTS API connection test failed: {e}")
            return False
