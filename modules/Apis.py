import websockets
import asyncio
import json
import random
import string
import requests
import time
import base64
import io
import soundfile
import re


GRADIO_FN = (
    29  # This is the index of the function in Text-Generation-Webui on commit ff610b4
)


class TGWapi(object):
    def __init__(self, config):
        super().__init__()
        self.ws_url = config["hosts"]["TGWWS_api"]
        self.api_url = config["hosts"]["TGW_api"]
        self.rephrase_prompt = config["prompts"]["rephrase"]["vicuna"]
        self.rephrase2_prompt = config["prompts"]["rephrase2"]["vicuna"]
        self.ask_prompt = config["prompts"]["ask"]["alpaca"]
        self.params = config["params"]

    def random_hash():
        letters = string.ascii_lowercase + string.digits
        return "".join(random.choice(letters) for i in range(9))

    def create_form_data(self, prompt, params=None):
        if params is None:
            params = self.params

        payload = json.dumps([prompt, params])
        form_data = {"data": [payload]}
        return form_data

    async def run(self, text, params=None):
        payload = json.dumps([text, params])
        session = self.random_hash()

        async with websockets.connect(self.ws_url) as websocket:
            while content := json.loads(await websocket.recv()):
                match content["msg"]:
                    case "send_hash":
                        await websocket.send(
                            json.dumps({"session_hash": session, "fn_index": GRADIO_FN})
                        )
                    case "estimation":
                        pass
                    case "send_data":
                        await websocket.send(
                            json.dumps(
                                {
                                    "session_hash": session,
                                    "fn_index": GRADIO_FN,
                                    "data": [payload],
                                }
                            )
                        )
                    case "process_starts":
                        pass
                    case "process_generating" | "process_completed":
                        # remove prompt text from output
                        newtext = content["output"]["data"][0].replace(text, "")
                        yield newtext

                        # You can search for your desired end indicator and
                        #  stop generation by closing the websocket here
                        if content["msg"] == "process_completed":
                            break

    async def generate_stream(self, text, params=None):
        async for results in self.run(text, params):
            response = results
        return response

    def generate_stream_sync(self, text, params=None):
        return asyncio.run(self.generate_stream(text, params))

    def generate(self, text, params=None):
        request = {
            'prompt': text,
            'max_new_tokens': params['max_new_tokens'],
            'preset': 'simple-1'
        }
        
        response = requests.post(self.api_url, json=request)
        cleaned_text = response.json()["results"][0]["text"]
        cleaned_text = cleaned_text.replace(text, "")
        cleaned_text = cleaned_text.encode('ascii', 'ignore').decode('ascii')
        return cleaned_text

    def rephrase2(
        self,
        eddi_message,
        logs,
        max_new_tokens=None,
        do_sample=None,
        temperature=None,
        top_p=None,
        typical_p=None,
        repetition_penalty=None,
        encoder_repetition_penalty=None,
        top_k=None,
        min_length=None,
        no_repeat_ngram_size=None,
        num_beams=None,
        penalty_alpha=None,
        length_penalty=None,
        early_stopping=None,
        seed=None,
        add_bos_token=None,
        custom_stopping_strings=None,
        truncation_length=None,
        ban_eos_token=None,
        stream=False,
        debug=False,
    ):
        params = {
            "max_new_tokens": max_new_tokens or self.params["max_new_tokens"],
            "do_sample": do_sample or self.params["do_sample"],
            "temperature": temperature or self.params["temperature"],
            "top_p": top_p or self.params["top_p"],
            "typical_p": typical_p or self.params["typical_p"],
            "repetition_penalty": repetition_penalty
            or self.params["repetition_penalty"],
            "encoder_repetition_penalty": encoder_repetition_penalty
            or self.params["encoder_repetition_penalty"],
            "top_k": top_k or self.params["top_k"],
            "min_length": min_length or self.params["min_length"],
            "no_repeat_ngram_size": no_repeat_ngram_size
            or self.params["no_repeat_ngram_size"],
            "num_beams": num_beams or self.params["num_beams"],
            "penalty_alpha": penalty_alpha or self.params["penalty_alpha"],
            "length_penalty": length_penalty or self.params["length_penalty"],
            "early_stopping": early_stopping or self.params["early_stopping"],
            "seed": seed or self.params["seed"],
            "add_bos_token": add_bos_token or self.params["add_bos_token"],
            "custom_stopping_strings": custom_stopping_strings
            or self.params["custom_stopping_strings"],
            "truncation_length": truncation_length or self.params["truncation_length"],
            "ban_eos_token": ban_eos_token or self.params["ban_eos_token"],
        }
        # format logs message array to a string separated by newlines, and each line starting with a dash
        logs = "\n".join(["- " + log for log in logs])

        prompt = self.rephrase2_prompt.format(logs=logs, text=eddi_message)
        print(prompt)
        if debug:
            start = time.time()
            print("TGW API time: ", end="", flush=True)

        if stream:
            response = self.generate_stream_sync(prompt, params)
        else:
            response = self.generate(prompt, params)

        if debug:
            print(str(round(time.time() - start, 2)) + "s")

        return response.strip()
    
    def rephrase(
        self,
        eddi_message,
        max_new_tokens=None,
        do_sample=None,
        temperature=None,
        top_p=None,
        typical_p=None,
        repetition_penalty=None,
        encoder_repetition_penalty=None,
        top_k=None,
        min_length=None,
        no_repeat_ngram_size=None,
        num_beams=None,
        penalty_alpha=None,
        length_penalty=None,
        early_stopping=None,
        seed=None,
        add_bos_token=None,
        custom_stopping_strings=None,
        truncation_length=None,
        ban_eos_token=None,
        stream=False,
        debug=False,
    ):
        params = {
            "max_new_tokens": max_new_tokens or self.params["max_new_tokens"],
            "do_sample": do_sample or self.params["do_sample"],
            "temperature": temperature or self.params["temperature"],
            "top_p": top_p or self.params["top_p"],
            "typical_p": typical_p or self.params["typical_p"],
            "repetition_penalty": repetition_penalty
            or self.params["repetition_penalty"],
            "encoder_repetition_penalty": encoder_repetition_penalty
            or self.params["encoder_repetition_penalty"],
            "top_k": top_k or self.params["top_k"],
            "min_length": min_length or self.params["min_length"],
            "no_repeat_ngram_size": no_repeat_ngram_size
            or self.params["no_repeat_ngram_size"],
            "num_beams": num_beams or self.params["num_beams"],
            "penalty_alpha": penalty_alpha or self.params["penalty_alpha"],
            "length_penalty": length_penalty or self.params["length_penalty"],
            "early_stopping": early_stopping or self.params["early_stopping"],
            "seed": seed or self.params["seed"],
            "add_bos_token": add_bos_token or self.params["add_bos_token"],
            "custom_stopping_strings": custom_stopping_strings
            or self.params["custom_stopping_strings"],
            "truncation_length": truncation_length or self.params["truncation_length"],
            "ban_eos_token": ban_eos_token or self.params["ban_eos_token"],
        }
        prompt = self.rephrase_prompt.format(text=eddi_message)

        if debug:
            start = time.time()
            print("TGW API time: ", end="", flush=True)

        if stream:
            response = self.generate_stream_sync(prompt, params)
        else:
            response = self.generate(prompt, params)

        if debug:
            print(str(round(time.time() - start, 2)) + "s")

        return response.strip()

    def ask(
        self,
        question,
        context_messages,
        max_new_tokens=None,
        do_sample=None,
        temperature=None,
        top_p=None,
        typical_p=None,
        repetition_penalty=None,
        encoder_repetition_penalty=None,
        top_k=None,
        min_length=None,
        no_repeat_ngram_size=None,
        num_beams=None,
        penalty_alpha=None,
        length_penalty=None,
        early_stopping=None,
        seed=None,
        add_bos_token=None,
        custom_stopping_strings=None,
        truncation_length=None,
        ban_eos_token=None,
        debug=False,
    ):
        params = {
            "max_new_tokens": max_new_tokens or self.params["max_new_tokens"],
            "do_sample": do_sample or self.params["do_sample"],
            "temperature": temperature or self.params["temperature"],
            "top_p": top_p or self.params["top_p"],
            "typical_p": typical_p or self.params["typical_p"],
            "repetition_penalty": repetition_penalty
            or self.params["repetition_penalty"],
            "encoder_repetition_penalty": encoder_repetition_penalty
            or self.params["encoder_repetition_penalty"],
            "top_k": top_k or self.params["top_k"],
            "min_length": min_length or self.params["min_length"],
            "no_repeat_ngram_size": no_repeat_ngram_size
            or self.params["no_repeat_ngram_size"],
            "num_beams": num_beams or self.params["num_beams"],
            "penalty_alpha": penalty_alpha or self.params["penalty_alpha"],
            "length_penalty": length_penalty or self.params["length_penalty"],
            "early_stopping": early_stopping or self.params["early_stopping"],
            "seed": seed or self.params["seed"],
            "add_bos_token": add_bos_token or self.params["add_bos_token"],
            "custom_stopping_strings": custom_stopping_strings
            or self.params["custom_stopping_strings"],
            "truncation_length": truncation_length or self.params["truncation_length"],
            "ban_eos_token": ban_eos_token or self.params["ban_eos_token"],
        }

        # make role uppercase
        context_messages = list(
            map(
                lambda message: f'{message["role"].upper()}: "{message["text"]}"',
                context_messages,
            )
        )
        context_messages = list(
            map(lambda message: message[0].upper() + message[1:], context_messages)
        )
        context_messages = "\n".join(context_messages)

        prompt = self.ask_prompt.format(context=context_messages, question=question)

        if debug:
            start = time.time()
            print("TGW API time: ", end="", flush=True)

        response = self.generate(prompt, params)

        if debug:
            print(str(round(time.time() - start, 2)) + "s")
        cleaned_text = response.split('"')[1].strip()

        return cleaned_text


class TTSapi(object):
    def __init__(self, config):
        super().__init__()
        self.api_url = config["hosts"]["TTS_api"]
        self.sid1 = config["tts"]["sid1"]
        self.sid2 = config["tts"]["sid2"]
        from modules.AudioPlayer import AudioPlayer

        self.AudioPlayer = AudioPlayer(config)

    def record(self, sample_rate=44100, time=5):
        return self.AudioPlayer.record(sample_rate, time)
    
    def just_play(self, audio, sr):
        self.AudioPlayer.play(audio, sr)

    def generate(self, text, play=True, debug=False):
        form_data = {"text": text, "sid1": self.sid1, "sid2": self.sid2}
        if debug:
            start = time.time()
            print("TTS API time: ", end="", flush=True)
        response = requests.post(self.api_url, data=form_data)

        file_object = io.BytesIO(
            base64.b64decode(response.json()["audio"].encode("utf-8"))
        )
        file_object.seek(0)

        audio, sr = soundfile.read(file_object, dtype="float32")

        if debug:
            print(str(round(time.time() - start, 2)) + "s")
            soundfile.write("tmp/TTSapi_debug.wav", audio, sr, format="wav")

        if play:
            self.AudioPlayer.play(audio, sr)
        else:
            return audio, sr
