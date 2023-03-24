# <img src='https://camo.githubusercontent.com/57d5fd32c5b51e73fce9077a45f155db3edecd5dfe31d272d73569cb23ef779c/68747470733a2f2f692e696d6775722e636f6d2f6c41645a6a376d2e6a706567' card_color='#40DBB0' width='50' height='50' style='vertical-align:bottom'/> LlamaCPP Persona
 
Give OpenVoiceOS some sass with [LLaMA](https://arxiv.org/abs/2302.13971) model in [pure C/C++](https://github.com/ggerganov/llama.cpp)

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
