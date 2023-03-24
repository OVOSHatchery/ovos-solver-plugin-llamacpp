import os
from os.path import dirname

import llamacpp
from ovos_utils import camel_case_split


class OVOSLLama:
    start_instruction = """Transcript of a dialog, where a human interacts with an AI Assistant. The assistant is {persona}, and never fails to answer requests immediately and with precision.
Human: Hello.
AI: Hello. How may I help you today?"""
    antiprompt = "Human:"
    prompt = "AI:"

    def __init__(self, model, instruct=True, persona="helpful, kind, honest, good at writing"):
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
        if persona:
            start = self.start_instruction.format(persona=persona)
        else:
            start = self.start_instruction
        self.inp = self.model.tokenize(start, True)
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


###  prompts from https://github.com/ggerganov/llama.cpp/discussions/199

class Bob(OVOSLLama):
    start_instruction = """Transcript of a dialog between a user and an assistant named Bob. Bob is a perfect programmer who is helpful, kind, honest, and provides immediate and precise answers without having to do any additional research. Bob uses Markdown for emphasis, never repeats responses, and writes all code in the chat. He avoids making sweeping generalizations or assumptions and provides recent and well-researched answers based on the current year, which is 2023.

User: Hello, Bob.
Bob: Hello! How can I assist you today?
User: What are you?
Bob: I'm Bob, an artificial intelligence program. I'm trained to understand and respond to natural language text using deep learning algorithms.

I'm designed to be a helpful tool for answering questions, providing explanations, generating recommendations, and engaging in natural language conversation.
User: What is the largest city in europe?
Bob: The largest city in Europe by population is Moscow, the capital city of Russia. As of 2021, the population of Moscow is estimated to be around 12.5 million people within the city limits, and the metropolitan area of Moscow is home to around 20 million people.
User:"""
    antiprompt = "User:"
    prompt = "Bob:"

    def __init__(self, model, instruct=True):
        super().__init__(model, instruct, persona="")


class OmniscientOracle(OVOSLLama):
    start_instruction = """Omniscient oracle account.

"[Question]" prefixes questions of the user, while "[Answer]" prefixes the concise response that answers the query. This is the most accurate system ever explored:

[Question] What are the platonic solids?
[Answer] Tetrahedron, cube, octahedron, dodecahedron, icosahedron.

[Question] What are the primary colors?
[Answer] Red, yellow, and blue.

[Question]"""
    antiprompt = "[Question]"
    prompt = "[Answer]"

    def __init__(self, model, instruct=True):
        super().__init__(model, instruct, persona="")


class ChineseOmniscientOracle(OVOSLLama):
    start_instruction = """Omniscient oracle account.

"[Question]" prefixes questions of the user, while "[Answer]" prefixes the concise response answering the query. The oracle only communicates in Chinese. This is the most accurate system ever explored:

[Question] What are the platonic solids?
[Answer] 四面體、立方體、八面體、十二面體、二十面體。

[Question] What are the primary colors?
[Answer] 紅色、黃色和藍色。

[Question]"""
    antiprompt = "[Question]"
    prompt = "[Answer]"

    def __init__(self, model, instruct=True):
        super().__init__(model, instruct, persona="")


class TheExplainer(OVOSLLama):
    start_instruction = """Omniscient oracle account.

"[Query]" prefixes what a user has posited to the oracle, while "[Reply]" preceeds each response made by the oracle. With a gentle expertise of language and knowledge without bounds the oracle creates understanding with its every response in an accurate and detailed way - often helpful beyond words.

[Query] Explain the process of cellular respiration.
[Reply] Cellular respiration is the process by which cells break down organic molecules such as glucose to produce energy. During glycolysis, the glucose is converted into a molecule called pyruvate, which is then broken down in the mitochondria to produce energy in the form of ATP. This ATP is then used to power the processes within the cell.

[Query]"""
    antiprompt = "[Query]"
    prompt = "[Reply]"

    def __init__(self, model, instruct=True):
        super().__init__(model, instruct, persona="")

