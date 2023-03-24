from neon_solvers import AbstractSolver
from os.path import dirname
from ovos_solver_llamacpp.personas import OVOSLLama, Bob, OmniscientOracle, TheExplainer


class LlamaCPPSolver(AbstractSolver):

    def __init__(self, config=None):
        super().__init__(name="LlamaCPP", priority=94, config=config,
                         enable_cache=False, enable_tx=True)
        checkpoint = self.config.get("model")
        persona = self.config.get("persona", "helpful, kind, honest, good at writing")
        persona = persona.lower()
        if persona == "explainer":
            self.model = TheExplainer(checkpoint)
        elif persona == "bob":
            self.model = Bob(checkpoint)
        elif persona == "omniscient oracle":
            self.model = OmniscientOracle(checkpoint)
        else:
            self.model = OVOSLLama(checkpoint, persona=persona)

    # officially exported Solver methods
    def get_spoken_answer(self, query, context=None):
        return self.model.ask(query)


if __name__ == "__main__":
    LLAMA_MODEL_FILE = f"/{dirname(dirname(__file__))}/models/ggml-model-q4_0.bin"

    bot = LlamaCPPSolver({"model": LLAMA_MODEL_FILE})

    sentence = bot.spoken_answer("Qual é o teu animal favorito?", {"lang": "pt-pt"})
    print(sentence)

    for q in ["Does god exist?",
              "what is the speed of light?",
              "what is the meaning of life?",
              "What is your favorite color?",
              "What is best in life?"]:
        a = bot.get_spoken_answer(q)
        print(q, a)
