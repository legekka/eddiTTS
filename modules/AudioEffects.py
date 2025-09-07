import io
import wave
import math
import numpy as np
from typing import Dict, Any, Optional
from pedalboard import Pedalboard, Chorus, Delay, Reverb, Distortion
import logging


class AudioEffects:
    """Audio effects processor using pedalboard for real-time audio enhancement"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize audio effects processor
        
        Args:
            config: Configuration dictionary with effects settings
        """
        self.config = config or {}
        self.effects_config = self.config.get("audio_effects", {})
        self.enabled = self.effects_config.get("enabled", False)
        self.logger = logging.getLogger(__name__)
        
        # Create pedalboard with configured effects
        self.pedalboard = self._create_pedalboard()
        
        if self.enabled:
            self.logger.info("Audio effects initialized")
            self._log_effects_config()
        else:
            self.logger.info("Audio effects disabled")
    
    def _create_pedalboard(self) -> Pedalboard:
        """Create pedalboard with configured effects"""
        board = Pedalboard()
        
        if not self.enabled:
            return board
        
        # Add effects in order: Distortion -> Chorus -> Delay -> Reverb
        # This order provides the most natural sound progression
        
        # 1. Distortion (applied first for warmth/character)
        if self.effects_config.get("distortion", {}).get("enabled", False):
            distortion_config = self.effects_config["distortion"]
            board.append(Distortion(
                drive_db=distortion_config.get("drive_db", 10.0),
            ))
        
        # 2. Chorus (modulation effects)
        if self.effects_config.get("chorus", {}).get("enabled", False):
            chorus_config = self.effects_config["chorus"]
            board.append(Chorus(
                rate_hz=chorus_config.get("rate_hz", 1.0),
                depth=chorus_config.get("depth", 0.25),
                centre_delay_ms=chorus_config.get("centre_delay_ms", 7.0),
                feedback=chorus_config.get("feedback", 0.0),
                mix=chorus_config.get("mix", 0.5)
            ))
        
        # 3. Delay (echo effects)
        if self.effects_config.get("delay", {}).get("enabled", False):
            delay_config = self.effects_config["delay"]
            board.append(Delay(
                delay_seconds=delay_config.get("delay_seconds", 0.25),
                feedback=delay_config.get("feedback", 0.3),
                mix=delay_config.get("mix", 0.2)
            ))
        
        # 4. Reverb (spatial effects, applied last)
        if self.effects_config.get("reverb", {}).get("enabled", False):
            reverb_config = self.effects_config["reverb"]
            board.append(Reverb(
                room_size=reverb_config.get("room_size", 0.5),
                damping=reverb_config.get("damping", 0.5),
                wet_level=reverb_config.get("wet_level", 0.3),
                dry_level=reverb_config.get("dry_level", 0.8),
                width=reverb_config.get("width", 1.0),
                freeze_mode=reverb_config.get("freeze_mode", 0.0)
            ))
        
        return board
    
    def _log_effects_config(self):
        """Log the current effects configuration"""
        active_effects = []
        
        for effect_name in ["distortion", "chorus", "delay", "reverb"]:
            if self.effects_config.get(effect_name, {}).get("enabled", False):
                active_effects.append(effect_name.capitalize())
        
        if active_effects:
            self.logger.info(f"Active effects: {', '.join(active_effects)}")
        else:
            self.logger.info("No effects enabled")
    
    def _calculate_tail_time(self) -> float:
        """Calculate the required tail time based on enabled effects
        
        Returns:
            Tail time in seconds needed for effects to fade naturally
        """
        if not self.enabled:
            return 0.0
        
        tail_time = 0.0
        
        # Delay effect needs time for echoes to fade
        if self.effects_config.get("delay", {}).get("enabled", False):
            delay_config = self.effects_config["delay"]
            delay_seconds = delay_config.get("delay_seconds", 0.25)
            feedback = delay_config.get("feedback", 0.3)
            
            # Calculate time for delay echoes to fade below -60dB
            # Each echo is reduced by (1-feedback), so we need log calculation
            if feedback > 0:
                # Time for echoes to decay to 1/1000 of original (-60dB)
                decay_cycles = math.log(0.001) / math.log(feedback)
                delay_tail = delay_seconds * decay_cycles
            else:
                # Single echo only
                delay_tail = delay_seconds
            
            tail_time = max(tail_time, delay_tail)
        
        # Reverb effect needs time for reflections to fade
        if self.effects_config.get("reverb", {}).get("enabled", False):
            reverb_config = self.effects_config["reverb"]
            room_size = reverb_config.get("room_size", 0.5)
            damping = reverb_config.get("damping", 0.5)
            
            # Estimate reverb decay time based on room size and damping
            # Larger rooms and less damping = longer decay
            base_decay = 0.5  # Base decay time in seconds
            size_factor = 1.0 + (room_size * 2.0)  # Room size multiplier
            damping_factor = 2.0 - damping  # Less damping = longer decay
            
            reverb_tail = base_decay * size_factor * damping_factor
            tail_time = max(tail_time, reverb_tail)
        
        # Add a small buffer (10% extra)
        tail_time *= 1.1
        
        # Cap maximum tail time to prevent excessive audio length
        tail_time = min(tail_time, 3.0)  # Maximum 3 seconds of tail
        
        return tail_time
    
    def _add_silence_padding(self, audio_array: np.ndarray, sample_rate: int, tail_time: float) -> np.ndarray:
        """Add silence padding to the end of audio for effect tails
        
        Args:
            audio_array: Input audio array
            sample_rate: Audio sample rate
            tail_time: Tail time in seconds
            
        Returns:
            Audio array with silence padding added
        """
        if tail_time <= 0:
            return audio_array
        
        # Calculate number of samples for tail
        tail_samples = int(tail_time * sample_rate)
        
        if len(audio_array.shape) == 1:
            # Mono audio
            silence = np.zeros(tail_samples, dtype=audio_array.dtype)
            return np.concatenate([audio_array, silence])
        else:
            # Stereo audio (channels, samples)
            num_channels = audio_array.shape[0]
            silence = np.zeros((num_channels, tail_samples), dtype=audio_array.dtype)
            return np.concatenate([audio_array, silence], axis=1)
        
    def process_audio(self, audio_data: bytes) -> bytes:
        """Process audio data with configured effects
        
        Args:
            audio_data: Input WAV audio data as bytes
            
        Returns:
            Processed audio data as bytes
        """
        if not self.enabled or len(self.pedalboard) == 0:
            # No effects enabled, return original audio
            return audio_data
        
        try:
            # Parse input WAV data
            audio_io = io.BytesIO(audio_data)
            with wave.open(audio_io, 'rb') as wav_file:
                # Get audio parameters
                sample_rate = wav_file.getframerate()
                num_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                num_frames = wav_file.getnframes()
                
                # Read audio data as numpy array
                raw_audio = wav_file.readframes(num_frames)
                
                # Convert to numpy array
                if sample_width == 1:
                    audio_array = np.frombuffer(raw_audio, dtype=np.uint8)
                    audio_array = (audio_array.astype(np.float32) - 128) / 128.0
                elif sample_width == 2:
                    audio_array = np.frombuffer(raw_audio, dtype=np.int16)
                    audio_array = audio_array.astype(np.float32) / 32768.0
                elif sample_width == 4:
                    audio_array = np.frombuffer(raw_audio, dtype=np.int32)
                    audio_array = audio_array.astype(np.float32) / 2147483648.0
                else:
                    self.logger.warning(f"Unsupported sample width: {sample_width}")
                    return audio_data
                
                # Reshape for stereo/mono
                if num_channels == 2:
                    audio_array = audio_array.reshape(-1, 2).T
                elif num_channels == 1:
                    audio_array = audio_array.reshape(1, -1)
                else:
                    self.logger.warning(f"Unsupported channel count: {num_channels}")
                    return audio_data
                
                # Calculate required tail time and add padding
                tail_time = self._calculate_tail_time()
                if tail_time > 0:
                    self.logger.debug(f"Adding {tail_time:.2f}s tail for effects")
                    audio_array = self._add_silence_padding(audio_array, sample_rate, tail_time)
                
                # Apply effects
                original_samples = num_frames
                total_samples = audio_array.shape[1] if len(audio_array.shape) > 1 else len(audio_array)
                self.logger.debug(f"Processing audio: {sample_rate}Hz, {num_channels}ch, {original_samples}→{total_samples} samples")
                processed_audio = self.pedalboard(audio_array, sample_rate)
                
                # Convert back to original format
                if num_channels == 1:
                    processed_audio = processed_audio.flatten()
                else:
                    processed_audio = processed_audio.T.flatten()
                
                # Convert back to integer format
                if sample_width == 1:
                    processed_audio = ((processed_audio * 128.0) + 128).astype(np.uint8)
                elif sample_width == 2:
                    processed_audio = (processed_audio * 32767.0).astype(np.int16)
                elif sample_width == 4:
                    processed_audio = (processed_audio * 2147483647.0).astype(np.int32)
                
                # Create output WAV
                output_io = io.BytesIO()
                with wave.open(output_io, 'wb') as output_wav:
                    output_wav.setnchannels(num_channels)
                    output_wav.setsampwidth(sample_width)
                    output_wav.setframerate(sample_rate)
                    output_wav.writeframes(processed_audio.tobytes())
                
                return output_io.getvalue()
                
        except Exception as e:
            self.logger.error(f"Error processing audio effects: {e}")
            # Return original audio on error
            return audio_data
    
    def get_effects_info(self) -> Dict[str, Any]:
        """Get information about current effects configuration"""
        if not self.enabled:
            return {"enabled": False, "effects": [], "tail_time": 0.0}
        
        effects_info = []
        for effect in self.pedalboard:
            effect_type = type(effect).__name__
            effects_info.append({
                "type": effect_type,
                "parameters": str(effect)
            })
        
        tail_time = self._calculate_tail_time()
        
        return {
            "enabled": True,
            "effects_count": len(self.pedalboard),
            "effects": effects_info,
            "tail_time": tail_time
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update effects configuration and rebuild pedalboard
        
        Args:
            new_config: New configuration dictionary
        """
        self.config = new_config
        self.effects_config = self.config.get("audio_effects", {})
        self.enabled = self.effects_config.get("enabled", False)
        
        # Rebuild pedalboard with new config
        self.pedalboard = self._create_pedalboard()
        
        self.logger.info("Audio effects configuration updated")
        if self.enabled:
            self._log_effects_config()


if __name__ == "__main__":
    # Test the audio effects processor
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
    test_config = {
        "audio_effects": {
            "enabled": True,
            "reverb": {
                "enabled": True,
                "room_size": 0.7,
                "damping": 0.5,
                "wet_level": 0.3,
                "dry_level": 0.8,
                "width": 1.0,
                "freeze_mode": 0.0
            },
            "chorus": {
                "enabled": True,
                "rate_hz": 1.5,
                "depth": 0.3,
                "centre_delay_ms": 7.0,
                "feedback": 0.0,
                "mix": 0.4
            },
            "delay": {
                "enabled": False,
                "delay_seconds": 0.25,
                "feedback": 0.3,
                "mix": 0.2
            },
            "distortion": {
                "enabled": False,
                "drive_db": 10.0
            }
        }
    }
    
    print("Testing Audio Effects...")
    
    try:
        effects = AudioEffects(test_config)
        print(f"Effects info: {effects.get_effects_info()}")
        
        # Test with a TTS file if it exists
        try:
            test_files = ["tmp/tts_msg_1.wav", "test_tts_output.wav"]
            test_file = None
            
            for file_path in test_files:
                try:
                    with open(file_path, "rb") as f:
                        test_file = file_path
                        break
                except FileNotFoundError:
                    continue
            
            if test_file:
                print(f"Processing test file: {test_file}")
                with open(test_file, "rb") as f:
                    original_audio = f.read()
                
                processed_audio = effects.process_audio(original_audio)
                
                # Save processed audio
                output_file = "tmp/test_effects_output.wav"
                with open(output_file, "wb") as f:
                    f.write(processed_audio)
                
                print(f"Processed audio saved to: {output_file}")
                print(f"Original size: {len(original_audio)} bytes")
                print(f"Processed size: {len(processed_audio)} bytes")
                
            else:
                print("No test audio file found")
                
        except Exception as e:
            print(f"Error testing audio file: {e}")
            
    except Exception as e:
        print(f"Error testing audio effects: {e}")
