import os
import time
import json
import sys
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import queue
import threading
from modules.llmbackend import LlmBackend
from modules.TTSClient import VibeVoiceTTSClient


@dataclass
class Message:
    """Data class for messages in the processing pipeline"""
    id: str
    original_text: str
    rephrased_text: Optional[str] = None
    timestamp: float = None
    status: str = "pending"  # pending, rephrasing, rephrased, displaying, displayed, error
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class EddiTTSProcessorAsync:
    """Asynchronous processor for monitoring EDDI files and rephrasing content"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the processor with configuration"""
        self.config = self.load_config(config_path)
        self.messages: List[Dict] = []
        self.lines_processed = 0
        self.running = False
        
        # Message counters for unique IDs
        self.message_counter = 0
        
        # Queues for async processing
        self.file_to_llm_queue = queue.Queue()      # File monitor -> LLM worker
        self.llm_to_gui_queue = queue.Queue()       # LLM worker -> GUI worker
        self.llm_to_tts_queue = queue.Queue()       # LLM worker -> TTS worker (future)
        
        # Worker threads
        self.workers = []
        
        # Initialize GUI if enabled
        self.app = None
        self.gui = None
        if self.config.get("gui", False):
            try:
                self.init_gui()
            except Exception as e:
                print(f"GUI initialization failed: {e}")
                print("Continuing without GUI...")
                self.gui = None
        
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
        
        # Load system prompt
        self.system_prompt = self.load_system_prompt()
        
        # Set up speechresponder file path
        self.speechresponder_path = os.path.join(
            os.getenv("APPDATA", ""), "EDDI", "speechresponder.out"
        )
        
        # Load existing messages
        self.load_messages()
        
        print(f"EddiTTS Async Processor initialized")
        print(f"Monitoring: {self.speechresponder_path}")
        print(f"LLM Backend: {self.llm_backend.base_url}")
        print(f"Available models: {', '.join(self.llm_backend.list_models())}")
        if self.gui:
            print(f"GUI: Enabled")
        if self.tts_client:
            print(f"TTS: Enabled ({self.tts_client.base_url})")
        else:
            print(f"TTS: Disabled")
        print("-" * 50)
    
    def init_gui(self) -> None:
        """Initialize the GUI if PySide6 is available"""
        try:
            # Set Qt platform plugin for Windows
            import os
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
            
            from PySide6.QtWidgets import QApplication
            from modules.ImprovedGui import ImprovedGui
            
            if not QApplication.instance():
                self.app = QApplication(sys.argv)
            else:
                self.app = QApplication.instance()
            
            self.gui = ImprovedGui()
            self.gui.show()  # Make the GUI visible!
            print("GUI initialized successfully")
            
        except ImportError:
            print("PySide6 not available. GUI disabled.")
        except Exception as e:
            print(f"Error initializing GUI: {e}")
            raise
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using defaults.")
            return {"gui": False}
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
            
            start_time = time.time()
            
            response = self.llm_backend.chat(
                messages=messages,
                model=model,
                reasoning_effort="low",
                temperature=1,
                top_p=1,
            )
            
            api_time = time.time() - start_time
            
            rephrased = response.choices[0].message.content.strip()
            return rephrased
            
        except Exception as e:
            print(f"Error rephrasing text: {e}")
            print("Using original text.")
            return text
    
    def file_monitor_worker(self):
        """Worker thread that monitors the speechresponder.out file"""
        print("🔍 File monitor worker started")
        
        while self.running:
            try:
                if not os.path.exists(self.speechresponder_path):
                    time.sleep(0.5)
                    continue
                
                with open(self.speechresponder_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Initialize line count on first run
                if self.lines_processed == 0:
                    self.lines_processed = len(lines)
                    print(f"Initialized with {self.lines_processed} existing lines")
                    time.sleep(0.5)
                    continue
                
                # Process new lines
                if len(lines) > self.lines_processed:
                    new_lines_count = len(lines) - self.lines_processed
                    print(f"\\n🔍 {new_lines_count} new line(s) detected")
                    
                    for i in range(self.lines_processed, len(lines)):
                        original_text = lines[i].strip()
                        if original_text:  # Skip empty lines
                            # Create message object and add to LLM queue
                            self.message_counter += 1
                            message = Message(
                                id=f"msg_{self.message_counter}",
                                original_text=original_text
                            )
                            
                            print(f"📤 [{message.id}] Original: {original_text}")
                            
                            # Add to LLM processing queue
                            self.file_to_llm_queue.put(message)
                    
                    self.lines_processed = len(lines)
                
            except Exception as e:
                print(f"Error in file monitor: {e}")
            
            time.sleep(0.25)  # 250ms polling interval
        
        print("🔍 File monitor worker stopped")
    
    def llm_worker(self):
        """Worker thread that processes LLM rephrasing requests"""
        print("🧠 LLM worker started")
        
        while self.running:
            try:
                # Get message from queue (non-blocking)
                try:
                    message = self.file_to_llm_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                print(f"🧠 [{message.id}] Starting rephrasing...")
                message.status = "rephrasing"
                
                start_time = time.time()
                message.rephrased_text = self.rephrase_text(message.original_text)
                api_time = time.time() - start_time
                
                message.status = "rephrased"
                
                print(f"✨ [{message.id}] Rephrased in {api_time:.2f}s: {message.rephrased_text}")
                
                # Add to GUI queue
                self.llm_to_gui_queue.put(message)
                
                # Add to TTS queue for future use
                self.llm_to_tts_queue.put(message)
                
                # Log the message
                self.log_message(message.rephrased_text, "assistant")
                
                # Mark task as done
                self.file_to_llm_queue.task_done()
                
            except Exception as e:
                print(f"Error in LLM worker: {e}")
                if 'message' in locals():
                    message.status = "error"
                    self.file_to_llm_queue.task_done()
        
        print("🧠 LLM worker stopped")
    
    def process_gui_queue(self):
        """Process GUI queue in main thread (called from main loop)"""
        if not self.gui:
            return
        
        # Process all pending GUI messages (non-blocking)
        while True:
            try:
                message = self.llm_to_gui_queue.get_nowait()
                
                print(f"🖥️ [{message.id}] Displaying message...")
                message.status = "displaying"
                
                # Display in GUI - this will block until the entire animation sequence is complete
                # The ImprovedGui.display_message() starts the animation and returns immediately
                # Then gui.wait() blocks until all animations (fade-in, typing, display, fade-out) complete
                self.gui.display_message(message.rephrased_text)
                self.gui.wait()  # This blocks until the full display cycle is done
                
                message.status = "displayed"
                print(f"🖥️ [{message.id}] Display complete")
                
                # Mark task as done
                self.llm_to_gui_queue.task_done()
                
            except queue.Empty:
                # No more messages to process
                break
            except Exception as e:
                print(f"Error processing GUI queue: {e}")
                if 'message' in locals():
                    message.status = "error"
                    self.llm_to_gui_queue.task_done()
    
    def tts_worker(self):
        """Worker thread that handles TTS processing"""
        if not self.tts_client:
            print("🔊 TTS worker skipped - TTS not enabled")
            return
            
        print("🔊 TTS worker started")
        
        while self.running:
            try:
                # Get message from queue (non-blocking)
                try:
                    message = self.llm_to_tts_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                print(f"🔊 [{message.id}] Starting TTS generation...")
                
                # Get TTS configuration
                tts_config = self.config.get("tts", {})
                voice = tts_config.get("voice", "en-Alice_woman")
                cfg_scale = tts_config.get("cfg_scale", 1.3)
                
                try:
                    # Generate TTS audio (synchronous for this API)
                    start_time = time.time()
                    response = self.tts_client.generate_speech(
                        text=message.rephrased_text,
                        voice=voice,
                        cfg_scale=cfg_scale
                    )
                    tts_time = time.time() - start_time
                    
                    if response.status == "completed" and response.audio_data:
                        # Save audio file
                        audio_filename = f"tmp/tts_{message.id}.wav"
                        os.makedirs("tmp", exist_ok=True)
                        self.tts_client.save_audio_to_file(response.audio_data, audio_filename)
                        
                        audio_size_kb = len(response.audio_data) / 1024
                        print(f"🔊 [{message.id}] TTS completed in {tts_time:.2f}s - {audio_size_kb:.1f}KB saved to {audio_filename}")
                        
                        # TODO: In future, could play the audio here or add to audio playback queue
                        # TODO: Could also calculate actual audio duration and update GUI timing
                        
                    else:
                        print(f"🔊 [{message.id}] TTS generation failed: {response.error_message}")
                    
                except Exception as e:
                    print(f"🔊 [{message.id}] TTS generation failed: {e}")
                
                # Mark task as done
                self.llm_to_tts_queue.task_done()
                
            except Exception as e:
                print(f"Error in TTS worker: {e}")
                if 'message' in locals():
                    self.llm_to_tts_queue.task_done()
        
        print("🔊 TTS worker stopped")
    
    def start_workers(self):
        """Start all worker threads"""
        self.running = True
        
        # Create and start worker threads (excluding GUI worker)
        workers_config = [
            ("file_monitor", self.file_monitor_worker),
            ("llm_processor", self.llm_worker),
            ("tts_processor", self.tts_worker),
        ]
        
        # Note: GUI processing happens in main thread, not a separate worker
        
        for name, worker_func in workers_config:
            worker = threading.Thread(target=worker_func, name=name, daemon=True)
            worker.start()
            self.workers.append(worker)
        
        print(f"✅ Started {len(self.workers)} worker threads")
        if self.gui:
            print("🖥️ GUI processing will run in main thread")
    
    def stop_workers(self):
        """Stop all worker threads"""
        print("\\n🛑 Stopping workers...")
        self.running = False
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                print(f"⚠️ Worker {worker.name} did not stop gracefully")
        
        # Clean up TTS client
        if self.tts_client:
            self.tts_client.close()
        
        print("✅ All workers stopped")
    
    def run(self) -> None:
        """Main processing loop"""
        print("🚀 Starting EddiTTS async processor...")
        print("Press Ctrl+C to stop")
        if self.gui:
            print("💡 GUI is active - messages will be displayed on screen")
            print("🖥️ GUI processing runs in main thread for Qt compatibility")
        print("")
        
        try:
            # Start worker threads
            self.start_workers()
            
            # Main loop - process GUI events and handle GUI queue in main thread
            while True:
                # Process any pending GUI messages (if GUI is enabled)
                if self.gui:
                    # Process GUI queue in main thread
                    self.process_gui_queue()
                    # Process GUI events to keep it responsive
                    self.gui.sleep(50)  # 50ms - faster processing for responsiveness
                else:
                    time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\\n\\n🛑 Stopping EddiTTS processor...")
            self.stop_workers()
            print(f"Total messages processed: {len(self.messages)}")
            print("Goodbye, Commander! o7")


def main():
    """Main entry point"""
    try:
        processor = EddiTTSProcessorAsync()
        processor.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
