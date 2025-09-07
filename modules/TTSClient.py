import requests
import time
import uuid
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging


@dataclass
class TTSRequest:
    """Data class for TTS generation requests"""
    text: str
    voice: str = "en-Alice_woman"
    cfg_scale: float = 1.3
    task_id: Optional[str] = None
    
    def __post_init__(self):
        if self.task_id is None:
            self.task_id = str(uuid.uuid4())


@dataclass
class TTSResponse:
    """Data class for TTS generation responses"""
    task_id: str
    status: str  # queued, running, completed, failed
    queue_position: Optional[int] = None
    generation_time: Optional[float] = None
    audio_data: Optional[bytes] = None
    error_message: Optional[str] = None


class VibeVoiceTTSClient:
    """Client for VibeVoice FastAPI TTS service"""
    
    def __init__(self, base_url: str = "http://172.16.240.5:8500", timeout: int = 30):
        """
        Initialize the TTS client
        
        Args:
            base_url: Base URL of the VibeVoice FastAPI service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Test connection
        try:
            self.test_connection()
        except Exception as e:
            self.logger.warning(f"TTS service connection test failed: {e}")
    
    def test_connection(self) -> bool:
        """Test if the TTS service is available"""
        try:
            response = self.session.get(f"{self.base_url}/voices", timeout=5)
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.error(f"TTS service unavailable: {e}")
            return False
    
    def get_available_voices(self) -> List[str]:
        """Get list of available voices"""
        try:
            response = self.session.get(f"{self.base_url}/voices", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get voices: {e}")
            return ["en-Alice_woman"]  # Fallback to default
    
    def submit_tts_request(self, text: str, voice: str = "en-Alice_woman", cfg_scale: float = 1.3) -> TTSRequest:
        """
        Submit a TTS generation request (synchronous - returns audio directly)
        
        Args:
            text: Text to convert to speech (single speaker format)
            voice: Voice to use for generation
            cfg_scale: Configuration scale (1.0-2.0, higher = more expressive)
            
        Returns:
            TTSRequest object with task_id and audio_data
        """
        # Format text for single speaker
        script = f"Speaker 1: {text}"
        
        request_data = {
            "script": script,
            "speaker_names": [voice],
            "cfg_scale": cfg_scale
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/generate",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # This API returns audio directly, not JSON with task_id
            if response.headers.get('content-type', '').startswith('audio/'):
                # Generate a task ID for tracking
                task_id = str(uuid.uuid4())
                
                tts_request = TTSRequest(
                    text=text,
                    voice=voice,
                    cfg_scale=cfg_scale,
                    task_id=task_id
                )
                
                self.logger.info(f"TTS request completed synchronously: {task_id}")
                return tts_request, response.content
            else:
                # Try to parse as JSON (async API)
                result = response.json()
                task_id = result.get("task_id")
                
                if not task_id:
                    raise ValueError("No task_id received from TTS service")
                
                tts_request = TTSRequest(
                    text=text,
                    voice=voice,
                    cfg_scale=cfg_scale,
                    task_id=task_id
                )
                
                self.logger.info(f"TTS request submitted: {task_id}")
                return tts_request, None
            
        except Exception as e:
            self.logger.error(f"Failed to submit TTS request: {e}")
            raise
    
    def check_status(self, task_id: str) -> TTSResponse:
        """
        Check the status of a TTS generation job
        
        Args:
            task_id: Task ID returned from submit_tts_request
            
        Returns:
            TTSResponse with current status
        """
        try:
            response = self.session.get(f"{self.base_url}/status/{task_id}", timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            return TTSResponse(
                task_id=task_id,
                status=data.get("status", "unknown"),
                queue_position=data.get("queue_position"),
                generation_time=data.get("generation_time")
            )
            
        except Exception as e:
            self.logger.error(f"Failed to check status for {task_id}: {e}")
            return TTSResponse(
                task_id=task_id,
                status="failed",
                error_message=str(e)
            )
    
    def get_result(self, task_id: str) -> TTSResponse:
        """
        Get the audio result for a completed TTS job
        
        Args:
            task_id: Task ID of completed job
            
        Returns:
            TTSResponse with audio_data if successful
        """
        try:
            response = self.session.get(f"{self.base_url}/result/{task_id}", timeout=self.timeout)
            response.raise_for_status()
            
            return TTSResponse(
                task_id=task_id,
                status="completed",
                audio_data=response.content
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get result for {task_id}: {e}")
            return TTSResponse(
                task_id=task_id,
                status="failed",
                error_message=str(e)
            )
    
    def generate_speech(self, text: str, voice: str = "en-Alice_woman", 
                       cfg_scale: float = 1.3, max_wait_time: int = 300,
                       poll_interval: float = 2.0) -> TTSResponse:
        """
        Generate speech and wait for completion (blocking)
        
        Args:
            text: Text to convert to speech
            voice: Voice to use
            cfg_scale: Configuration scale
            max_wait_time: Maximum time to wait in seconds (unused for sync API)
            poll_interval: How often to check status in seconds (unused for sync API)
            
        Returns:
            TTSResponse with audio_data if successful
        """
        try:
            # Submit request (this now returns audio directly for sync API)
            request, audio_data = self.submit_tts_request(text, voice, cfg_scale)
            
            if audio_data:
                # Synchronous API - we have the audio immediately
                return TTSResponse(
                    task_id=request.task_id,
                    status="completed",
                    audio_data=audio_data
                )
            else:
                # Asynchronous API - need to poll for completion
                task_id = request.task_id
                start_time = time.time()
                
                while time.time() - start_time < max_wait_time:
                    # Check status
                    status_response = self.check_status(task_id)
                    
                    if status_response.status == "completed":
                        # Get the audio result
                        return self.get_result(task_id)
                    
                    elif status_response.status == "failed":
                        self.logger.error(f"TTS generation failed for task {task_id}")
                        return status_response
                    
                    elif status_response.status in ["queued", "running"]:
                        # Log progress
                        if status_response.queue_position is not None:
                            self.logger.info(f"TTS task {task_id}: {status_response.status} (queue position: {status_response.queue_position})")
                        else:
                            self.logger.info(f"TTS task {task_id}: {status_response.status}")
                        
                        time.sleep(poll_interval)
                    
                    else:
                        self.logger.warning(f"Unknown status for task {task_id}: {status_response.status}")
                        time.sleep(poll_interval)
                
                # Timeout
                self.logger.error(f"TTS generation timeout for task {task_id}")
                return TTSResponse(
                    task_id=task_id,
                    status="failed",
                    error_message=f"Timeout after {max_wait_time} seconds"
                )
                
        except Exception as e:
            self.logger.error(f"TTS generation failed: {e}")
            return TTSResponse(
                task_id="error",
                status="failed",
                error_message=str(e)
            )
    
    def generate_speech_async(self, text: str, voice: str = "en-Alice_woman", 
                             cfg_scale: float = 1.3) -> TTSResponse:
        """
        Generate speech asynchronously (returns immediately with result for sync API)
        
        Args:
            text: Text to convert to speech
            voice: Voice to use
            cfg_scale: Configuration scale
            
        Returns:
            TTSResponse object with audio data (for sync API) or TTSRequest for async API
        """
        try:
            request, audio_data = self.submit_tts_request(text, voice, cfg_scale)
            
            if audio_data:
                # Synchronous API - return completed response
                return TTSResponse(
                    task_id=request.task_id,
                    status="completed",
                    audio_data=audio_data
                )
            else:
                # Asynchronous API - return request for later polling
                return TTSResponse(
                    task_id=request.task_id,
                    status="submitted"
                )
                
        except Exception as e:
            self.logger.error(f"TTS async generation failed: {e}")
            return TTSResponse(
                task_id="error",
                status="failed",
                error_message=str(e)
            )
    
    def save_audio_to_file(self, audio_data: bytes, filepath: str) -> bool:
        """
        Save audio data to a file
        
        Args:
            audio_data: Audio bytes from TTS response
            filepath: Path to save the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'wb') as f:
                f.write(audio_data)
            self.logger.info(f"Audio saved to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save audio to {filepath}: {e}")
            return False
    
    def close(self):
        """Close the session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function for quick TTS generation
def generate_tts(text: str, voice: str = "en-Alice_woman", 
                base_url: str = "http://172.16.240.5:8500") -> bytes:
    """
    Quick function to generate TTS audio
    
    Args:
        text: Text to convert to speech
        voice: Voice to use
        base_url: TTS service URL
        
    Returns:
        Audio data as bytes
        
    Raises:
        Exception if generation fails
    """
    with VibeVoiceTTSClient(base_url) as client:
        response = client.generate_speech(text, voice)
        
        if response.status != "completed" or not response.audio_data:
            raise Exception(f"TTS generation failed: {response.error_message}")
        
        return response.audio_data


if __name__ == "__main__":
    # Test the TTS client
    logging.basicConfig(level=logging.INFO)
    
    print("Testing VibeVoice TTS Client...")
    
    try:
        with VibeVoiceTTSClient() as client:
            # Test connection
            voices = client.get_available_voices()
            print(f"Available voices: {voices}")
            
            # Test TTS generation
            test_text = "Hello Commander, this is a test of the VibeVoice TTS system."
            print(f"Generating speech for: {test_text}")
            
            response = client.generate_speech(test_text, voice="en-Alice_woman")
            
            if response.status == "completed" and response.audio_data:
                print(f"TTS generation successful! Audio size: {len(response.audio_data)} bytes")
                
                # Save to file for testing
                client.save_audio_to_file(response.audio_data, "test_tts_output.wav")
            else:
                print(f"TTS generation failed: {response.error_message}")
                
    except Exception as e:
        print(f"Error testing TTS client: {e}")
