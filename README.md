# <img src='https://camo.githubusercontent.com/0941c760a0702f80cbb8e16433afa1d0a76edfb82b6bae35271cf86a8c02e062/68747470733a2f2f692e6962622e636f2f6859666e31547a2f6c6c616d61637070336c6f772e6a7067' card_color='#40DBB0' width='50' height='50' style='vertical-align:bottom'/> LlamaCPP Persona
 
Give OpenVoiceOS some sass with [LLaMA](https://arxiv.org/abs/2302.13971) model in [pure C/C++](https://github.com/ggerganov/llama.cpp)

powered by [llamacpp-python](https://github.com/thomasantony/llamacpp-python)

## Examples 
* "What is best in life?"
* "Do you like dogs"
* "Does God exist?"


## Usage

Spoken answers api

```python
from ovos_solver_llamacpp import LlamaCPPSolver

LLAMA_MODEL_FILE = "./models/ggml-model-q4_0.bin"

# persona = "omniscient oracle" # hardcoded personas, "explainer"|"bob"|"omniscient oracle"
persona = "helpful, kind, honest, good at writing"  # description of assistant
bot = LlamaCPPSolver({"model": LLAMA_MODEL_FILE, 
                      "persona": persona})

sentence = bot.spoken_answer("Qual é o teu animal favorito?", {"lang": "pt-pt"})
# Meus animais favoritos são cães, gatos e tartarugas!

for q in ["Does god exist?",
          "what is the speed of light?",
          "what is the meaning of life?",
          "What is your favorite color?",
          "What is best in life?"]:
    a = bot.get_spoken_answer(q)
    print(q, a)
```
