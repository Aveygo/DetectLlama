# pip3 install git+https://github.com/huggingface/transformers
# pip3 install accelerate bitsandbytes

from transformers import AutoTokenizer, LlamaForCausalLM
import time, torch, re

class DetectLlama:
    def __init__(self):
        self.model = LlamaForCausalLM.from_pretrained("model_converted/llama-7b/", load_in_8bit=True, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained("model_converted/tokenizer/")

        self.max_length = 512
        self.stride = 51

    def clean_text(self, text:str):
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        text = text.strip()
        text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" :", ":").replace(" ;", ";")
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text

    def get_ppl(self, text:str):
        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        # Compute log-likelihood of the sentence
        nlls = []
        likelihoods = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                likelihoods.append(neg_log_likelihood)

            yield {
                "text": self.tokenizer.decode(encodings.input_ids[0, begin_loc:end_loc]),
                "score": int(torch.exp(neg_log_likelihood / trg_len)),
            }

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    def calculate(self, text:str):
        # Clean text
        text = self.clean_text(text)

        # Convert to sentences
        sentences = text.split(".")
        sentences = [sentence.strip() for sentence in sentences if sentence.strip() != ""]
        sentences = [sentence + "." for sentence in sentences]

        for sentence in sentences:
            yield from self.get_ppl(sentence)
