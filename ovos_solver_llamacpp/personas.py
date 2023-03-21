import os
from os.path import dirname

import llamacpp
from ovos_utils import camel_case_split


class OVOSLLama:
    start_instruction = """Transcript of a dialog, where a human interacts with an AI Assistant. The assistant  is helpful, kind, honest, good at writing, and never fails to answer requests immediately and with precision.
Human: Hello.
AI: Hello. How may I help you today?"""
    antiprompt = "Human:"
    prompt = "AI:"

    def __init__(self, model, instruct=True):
        # TODO - from config
        params = llamacpp.gpt_params(
            model,  # model,
            4096,  # ctx_size
            128,  # n_predict
            40,  # top_k
            0.95,  # top_p
            0.7,  # temp
            1.30,  # repeat_penalty
            # -1,  # seed
            666,
            os.cpu_count(),  # threads
            64,  # repeat_last_n
            8,  # batch_size
        )
        self.model = llamacpp.PyLLAMA(params)
        self.model.prepare_context()
        self.model.set_antiprompt(self.antiprompt)

        self.instruct = instruct
        self.inp = self.model.tokenize(self.start_instruction, True)
        self.inp_pfx = self.model.tokenize(f"\n\n{self.antiprompt}", True)
        self.inp_sfx = self.model.tokenize(f"\n\n{self.prompt}", False)

        self.model.add_bos()
        self.model.update_input_tokens(self.inp)

        self._1st = True

    def ask(self, utterance, early_stop=True):
        if not utterance:
            return "?"
        if not utterance.endswith(".") or not utterance.endswith("?"):
            utterance += "?" if utterance.startswith("wh") else "."

        ans = ""
        input_noecho = False
        is_interacting = True
        in_parantheses = False
        while not self.model.is_finished():
            if self.model.has_unconsumed_input():
                self.model.ingest_all_pending_input(not input_noecho)
                continue
            input_noecho = False

            if self.model.is_antiprompt_present():
                is_interacting = True

            if is_interacting:
                if self.instruct:
                    self.model.update_input_tokens(self.inp_pfx)
                self.model.update_input(utterance)
                if self.instruct:
                    self.model.update_input_tokens(self.inp_sfx)
                input_noecho = True
                is_interacting = False

            text, is_finished = self.model.infer_text()
            ans += text
            if not ans:
                continue

            if in_parantheses and ")" in text:
                in_parantheses = False
            elif "(" in text:
                in_parantheses = True

            bad_ends = any((ans.endswith(b) for b in [".", "!", "?", "\n"]))
            stop = all((len(ans), early_stop, not in_parantheses,
                        len(ans.split()) > 4, bad_ends))
            if is_finished or stop:
                self.model.ingest_all_pending_input(not input_noecho)
                break

        self.model.reset_remaining_tokens()
        return self._apply_text_hacks(ans)

    def _apply_text_hacks(self, ans):
        if ans.strip():
            # handle when llama continues with a made up user question
            if self.antiprompt:
                ans = ans.split(self.antiprompt)[0]

            # HACK: there seems to be a bug where output starts with a unrelated word???
            # sometimes followed by whitespace sometimes not
            wds = ans.split()
            # handle no whitespace case
            t = camel_case_split(wds[0]).split(" ")
            if len(t) == 2:
                wds[0] = t[-1]
            # handle whitespace case
            elif len(wds) > 1 and wds[1][0].isupper():
                wds[0] = ""
            ans = " ".join([w for w in wds if w])

            # llama 4B - bad starts, calling "user"
            bad_starts = ["("]
            for b in bad_starts:
                if ans.startswith(b):
                    ans = ans[len(b):]

            # llama 4B - bad ends, end token "\end{code}"
            bad_ends = ["\end{code}", self.prompt, self.antiprompt, " (1)"]
            for b in bad_ends:
                if ans.endswith(b):
                    ans = ans[:-1 * len(b)]

        return ans or "I don't known"


class OVOSAlpaca(OVOSLLama):
    start_instruction = """Below is an start_instruction that describes a task. Write a response that appropriately completes the request.\n\n"""
    antiprompt = "## Instruction:\n\n"
    prompt = "### Response:\n\n"

    def _apply_text_hacks(self, ans):
        if ans.strip():
            # handle when llama continues with a made up user question
            if self.antiprompt:
                ans = ans.split(self.antiprompt)[0]

            # HACK: there seems to be a bug where output starts with a unrelated word???
            # sometimes followed by whitespace sometimes not
            wds = ans.split()
            # handle no whitespace case
            t = camel_case_split(wds[0]).split(" ")
            if len(t) == 2:
                wds[0] = t[-1]
            # handle whitespace case
            elif len(wds) > 1 and wds[1][0].isupper():
                wds[0] = ""
            ans = " ".join([w for w in wds if w])

            bad_ends = [self.prompt, self.antiprompt]
            for b in bad_ends:
                if ans.endswith(b):
                    ans = ans[:-1 * len(b)]

            # with alpaca somethings answers start with "#"
            while ans[0] in ["#"]:
                ans = ans[1:]

        return ans or "I don't known"


if __name__ == "__main__":

    questions = ["tell me about space",
                 "tell me about God",
                 "Does God exist?",
                 "When will the world end?"]


    def tesr_llama():
        LLAMA_MODEL_FILE = f"/{dirname(__file__)}/models/ggml-model-q4_0.bin"
        llama = OVOSLLama(LLAMA_MODEL_FILE)
        for q in questions:
            answer = llama.ask(q)
            print("\nQ:", q, "\nA:", answer)


    def tesr_alpaca():
        ALPACA_MODEL_FILE = f"/{dirname(__file__)}/models/ggml-alpaca-7b-q4.bin"
        alpaca = OVOSAlpaca(ALPACA_MODEL_FILE)
        for q in questions:
            answer = alpaca.ask(q, early_stop=True)
            print("\nQ:", q, "\nA:", answer)


    tesr_alpaca()
