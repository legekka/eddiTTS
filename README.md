# eddiTTS

This is an AI-assisted extension for EDDI. It is designed to generate more natural-sounding speech not just by using a better TTS engine, but also by using AI to rephrase the text. Also provides interaction capabilities. It aims to be a drop-in replacement for the ingame COVAS, by being an AI assistant.

It even has a small GUI window that can be dropped into VR to see what the assistant is saying.

**Notice: This is using an older version of oobabooga's API for LLM interaction, and I have to update it to the current one.**

## Some gameplay footage from youtube

This is before the GUI was made.

[![Video](https://img.youtube.com/vi/ejV9PRwBa7g/maxresdefault.jpg)](https://youtu.be/ejV9PRwBa7g)

## Installation

I suggest using a virtual environment for this (conda or venv).
Also, I recommend using Python 3.9 or higher.

1. Clone this repository
2. Install the requirements: 
```
pip install -r requirements.txt
```
3. Create a `config.json` file in the root directory of this repository. See [config.json.example](config.json.example) for an example.

## Usage

To start the program, just run:
```
python launch.py
```
