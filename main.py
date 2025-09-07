import os
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from modules.llmbackend import LlmBackend
from modules.TTSClient import VibeVoiceTTSClient
from modules.AudioPlayer import AudioPlayer
from modules.AudioEffects import AudioEffects


class EddiTTSProcessor:
    """Main processor for monitoring EDDI files and rephrasing content"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the processor with configuration"""
        self.config = self.load_config(config_path)
        self.messages: List[Dict] = []
        self.lines_processed = 0
        
        # Initialize GUI if enabled
        self.app = None
        self.gui = None
        if self.config.get("gui", False):
            self.init_gui()
        
        # Initialize LLM backend
        self.llm_backend = LlmBackend(
            base_url=self.config.get("llm_backend", {}).get("base_url", "http://localhost:11434/v1"),
            api_key=self.config.get("llm_backend", {}).get("api_key", "valami")
        )
        
        # Initialize TTS client if enabled
        self.tts_client = None
        if self.config.get("tts", {}).get("enabled", False):
            try:
                self.tts_client = VibeVoiceTTSClient(
                    base_url=self.config["tts"].get("base_url", "http://172.16.240.5:8500"),
                    timeout=self.config["tts"].get("timeout", 30)
                )
                print(f"TTS Client initialized: {self.tts_client.base_url}")
            except Exception as e:
                print(f"TTS Client initialization failed: {e}")
                print("Continuing without TTS...")
                self.tts_client = None
        
        # Initialize Audio Player for TTS playback
        self.audio_player = None
        if self.tts_client:
            try:
                self.audio_player = AudioPlayer(self.config)
                print(f"Audio Player initialized: {self.audio_player.default_device['name'] if self.audio_player.default_device else 'No device'}")
            except Exception as e:
                print(f"Audio Player initialization failed: {e}")
                print("TTS audio will be saved but not played...")
                self.audio_player = None
        
        # Initialize Audio Effects processor
        self.audio_effects = None
        try:
            self.audio_effects = AudioEffects(self.config)
            effects_info = self.audio_effects.get_effects_info()
            if effects_info["enabled"]:
                tail_time = effects_info.get("tail_time", 0.0)
                print(f"Audio Effects initialized: {effects_info['effects_count']} effect(s) active, {tail_time:.2f}s tail time")
            else:
                print("Audio Effects: Disabled")
        except Exception as e:
            print(f"Audio Effects initialization failed: {e}")
            print("Continuing without audio effects...")
            self.audio_effects = None
        
        # Load system prompt
        self.system_prompt = self.load_system_prompt()
        
        # Set up speechresponder file path
        self.speechresponder_path = os.path.join(
            os.getenv("APPDATA", ""), "EDDI", "speechresponder.out"
        )
        
        # Load existing messages
        self.load_messages()
        
        print(f"EddiTTS Processor initialized")
        print(f"Monitoring: {self.speechresponder_path}")
        print(f"LLM Backend: {self.llm_backend.base_url}")
        print(f"Available models: {', '.join(self.llm_backend.list_models())}")
        if self.gui:
            print(f"GUI: Enabled")
        if self.tts_client:
            print(f"TTS: Enabled ({self.tts_client.base_url})")
            if self.audio_player:
                device_name = self.audio_player.default_device['name'] if self.audio_player.default_device else 'Unknown'
                print(f"Audio: Enabled ({device_name})")
                if self.audio_effects and self.audio_effects.enabled:
                    effects_info = self.audio_effects.get_effects_info()
                    active_effects = [effect["type"] for effect in effects_info["effects"]]
                    print(f"Audio Effects: Enabled ({', '.join(active_effects)})")
                else:
                    print(f"Audio Effects: Disabled")
            else:
                print(f"Audio: Disabled (TTS files will be saved only)")
        else:
            print(f"TTS: Disabled")
        print("-" * 50)
    
    def init_gui(self) -> None:
        """Initialize the GUI components"""
        try:
            from PySide6.QtWidgets import QApplication
            from modules.ImprovedGui import ImprovedGui
            
            # Create QApplication if it doesn't exist
            if not QApplication.instance():
                self.app = QApplication(sys.argv)
            else:
                self.app = QApplication.instance()
            
            self.gui = ImprovedGui()
            self.gui.show()
            print("Improved GUI initialized successfully")
        except Exception as e:
            print(f"Failed to initialize GUI: {e}")
            print("Continuing without GUI...")
            self.config["gui"] = False
            self.gui = None
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Add default LLM backend config if not present
            if "llm_backend" not in config:
                config["llm_backend"] = {
                    "base_url": "http://localhost:11434/v1",
                    "api_key": "valami",
                    "model": "gpt-oss:20b",
                    "context_messages": 5
                }
            
            return config
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default configuration.")
            return {
                "gui": False,
                "llm_backend": {
                    "base_url": "http://localhost:11434/v1",
                    "api_key": "valami",
                    "model": "gpt-oss:20b",
                    "context_messages": 5
                }
            }
        except json.JSONDecodeError as e:
            print(f"Error parsing config file: {e}")
            raise
    
    def load_system_prompt(self) -> str:
        """Load the system prompt for rephrasing"""
        prompt_path = "prompts/rephrase_openai.txt"
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Prompt file {prompt_path} not found. Using default prompt.")
            return ("You are a helpful AI assistant that rephrases system messages "
                   "to be more natural and conversational while maintaining their meaning.")
    
    def load_messages(self) -> None:
        """Load existing messages from JSON file"""
        messages_path = "data/messages.json"
        try:
            with open(messages_path, 'r', encoding='utf-8') as f:
                self.messages = json.load(f)
            print(f"Loaded {len(self.messages)} existing messages")
        except FileNotFoundError:
            print("No existing messages file found. Starting fresh.")
            self.messages = []
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
        except json.JSONDecodeError as e:
            print(f"Error parsing messages file: {e}")
            print("Starting with empty messages list.")
            self.messages = []
    
    def save_messages(self) -> None:
        """Save messages to JSON file"""
        messages_path = "data/messages.json"
        try:
            with open(messages_path, 'w', encoding='utf-8') as f:
                json.dump(self.messages, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving messages: {e}")
    
    def log_message(self, text: str, role: str = "assistant") -> None:
        """Log a message to the messages list and save to file"""
        message = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "role": role,
            "text": text.strip()
        }
        self.messages.append(message)
        self.save_messages()
    
    def rephrase_text(self, text: str) -> str:
        """Rephrase text using the LLM backend with conversation history"""
        try:
            # Start with system prompt
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add previous conversation history if configured and available
            context_count = self.config["llm_backend"].get("context_messages", 5)
            if context_count > 0 and self.messages:
                # Get the last N messages as context
                recent_messages = self.messages[-context_count:]
                for msg in recent_messages:
                    # Add previous messages as assistant messages to provide context
                    messages.append({
                        "role": "assistant", 
                        "content": msg["text"]
                    })
            
            # Add the current message to rephrase
            messages.append({"role": "user", "content": text})
            
            # Get model from config
            model = self.config["llm_backend"].get("model", "gpt-oss:20b")
            
            # Check if model exists
            available_models = self.llm_backend.list_models()
            if model not in available_models:
                print(f"Warning: Model '{model}' not available. Using first available model.")
                model = available_models[0] if available_models else "default"
            
            print(f"LLM API time: ", end="", flush=True)
            start_time = time.time()
            
            response = self.llm_backend.chat(
                messages=messages,
                model=model,
                reasoning_effort="low",
                temperature=1,
                top_p=1,
            )
            
            api_time = time.time() - start_time
            print(f"{api_time:.2f}s")
            
            rephrased = response.choices[0].message.content.strip()
            return rephrased
            
        except Exception as e:
            print(f"Error rephrasing text: {e}")
            print("Using original text.")
            return text
    
    def generate_tts_audio(self, text: str, message_id: str = None) -> Optional[bytes]:
        """
        Generate TTS audio for the given text
        
        Args:
            text: Text to convert to speech
            message_id: Optional message ID for file naming
            
        Returns:
            Audio data as bytes if successful, None otherwise
        """
        if not self.tts_client:
            return None
        
        try:
            # Get TTS configuration
            tts_config = self.config.get("tts", {})
            voice = tts_config.get("voice", "en-Alice_woman")
            cfg_scale = tts_config.get("cfg_scale", 1.3)
            
            print(f"🔊 TTS generation: ", end="", flush=True)
            start_time = time.time()
            
            # Generate TTS audio
            response = self.tts_client.generate_speech(
                text=text,
                voice=voice,
                cfg_scale=cfg_scale
            )
            
            tts_time = time.time() - start_time
            
            if response.status == "completed" and response.audio_data:
                # Optionally save audio file for debugging
                if message_id:
                    audio_filename = f"tmp/tts_{message_id}.wav"
                    os.makedirs("tmp", exist_ok=True)
                    self.tts_client.save_audio_to_file(response.audio_data, audio_filename)
                
                audio_size_kb = len(response.audio_data) / 1024
                print(f"{tts_time:.2f}s ({audio_size_kb:.1f}KB)")
                
                return response.audio_data
            else:
                print(f"failed ({response.error_message})")
                return None
                
        except Exception as e:
            print(f"failed ({e})")
            return None
    
    def check_for_new_lines(self) -> None:
        """Check for new lines in the speechresponder.out file"""
        try:
            if not os.path.exists(self.speechresponder_path):
                return
            
            with open(self.speechresponder_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Initialize line count on first run
            if self.lines_processed == 0:
                self.lines_processed = len(lines)
                print(f"Initialized with {self.lines_processed} existing lines")
                return
            
            # Process new lines
            if len(lines) > self.lines_processed:
                new_lines_count = len(lines) - self.lines_processed
                print(f"\\n🔍 {new_lines_count} new line(s) detected")
                
                for i in range(self.lines_processed, len(lines)):
                    original_text = lines[i].strip()
                    if original_text:  # Skip empty lines
                        message_id = f"msg_{int(time.time())}_{i}"
                        print(f"\\n📤 [{message_id}] Original: {original_text}")
                        
                        start_time = time.time()
                        rephrased_text = self.rephrase_text(original_text)
                        llm_time = time.time() - start_time
                        
                        print(f"✨ [{message_id}] Rephrased: {rephrased_text}")
                        
                        # Generate TTS audio if enabled
                        audio_data = None
                        if self.tts_client:
                            audio_data = self.generate_tts_audio(rephrased_text, message_id)
                            
                            # Apply audio effects if enabled
                            if audio_data and self.audio_effects and self.audio_effects.enabled:
                                print(f"🎵 [{message_id}] Applying audio effects...", end="", flush=True)
                                effects_start_time = time.time()
                                audio_data = self.audio_effects.process_audio(audio_data)
                                effects_time = time.time() - effects_start_time
                                print(f" {effects_time:.3f}s")
                        
                        # Coordinate audio playback with GUI display
                        if audio_data and self.audio_player and self.gui:
                            # Play audio and start GUI when audio begins
                            print(f"🔊 [{message_id}] Starting coordinated audio + GUI display")
                            
                            # Get audio duration for better GUI timing coordination
                            audio_info = self.audio_player.get_audio_info(audio_data)
                            audio_duration_ms = int(audio_info.get('duration', 0) * 1000) if audio_info else None
                            
                            # Start GUI display immediately (in main thread)
                            print(f"🖥️ [{message_id}] Starting GUI display")
                            self.gui.display_message(rephrased_text, audio_duration_ms)
                            
                            def on_audio_start():
                                """Called when audio playback starts"""
                                print(f"🔊 [{message_id}] Audio playback started")
                            
                            def on_audio_complete():
                                """Called when audio playback completes"""
                                print(f"🔊 [{message_id}] Audio playback completed")
                            
                            # Play audio asynchronously (non-blocking) so GUI can animate smoothly
                            audio_thread = self.audio_player.play_audio_async(
                                audio_data, 
                                on_start=on_audio_start,
                                on_complete=on_audio_complete
                            )
                            
                            # Wait for GUI animation to complete
                            if self.gui:
                                self.gui.wait()
                            
                            # Ensure audio finishes before processing next message
                            audio_thread.join()
                            
                        elif self.gui:
                            # No audio - display GUI immediately (fallback behavior)
                            print(f"🖥️ [{message_id}] No audio - displaying GUI immediately")
                            self.gui.display_message(rephrased_text)
                            self.gui.wait()
                        
                        elif audio_data and self.audio_player:
                            # Audio only, no GUI - use async playback to avoid blocking
                            print(f"🔊 [{message_id}] Playing audio without GUI")
                            audio_thread = self.audio_player.play_audio_async(audio_data)
                            # Wait for audio to complete before processing next message
                            audio_thread.join()
                        
                        total_time = time.time() - start_time
                        print(f"⏱️ [{message_id}] Total time: {total_time:.2f}s")
                        
                        # Log the rephrased message
                        self.log_message(rephrased_text, "assistant")
                
                self.lines_processed = len(lines)
                print("-" * 50)
                
        except Exception as e:
            print(f"Error reading speechresponder file: {e}")
    
    def run(self) -> None:
        """Main processing loop"""
        print("🚀 Starting EddiTTS processor...")
        print("Press Ctrl+C to stop")
        if self.gui:
            print("💡 GUI is active - messages will be displayed on screen")
        if self.tts_client:
            print("🔊 TTS is active - audio files will be generated")
            if self.audio_effects and self.audio_effects.enabled:
                effects_info = self.audio_effects.get_effects_info()
                active_effects = [effect["type"] for effect in effects_info["effects"]]
                print(f"🎵 Audio effects are active: {', '.join(active_effects)}")
        print("")
        
        try:
            while True:
                self.check_for_new_lines()
                
                # Use GUI sleep if available, otherwise regular sleep
                if self.gui:
                    self.gui.sleep(250)  # 250ms
                else:
                    time.sleep(0.25)
                
        except KeyboardInterrupt:
            print("\\n\\n🛑 Stopping EddiTTS processor...")
            
            # Clean up audio player
            if self.audio_player:
                self.audio_player.close()
            
            # Clean up TTS client
            if self.tts_client:
                self.tts_client.close()
            
            print(f"Total messages processed: {len(self.messages)}")
            print("Goodbye, Commander! o7")


def main():
    """Main entry point"""
    try:
        processor = EddiTTSProcessor()
        processor.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
