# pip3 install git+https://github.com/huggingface/transformers
# pip3 install accelerate bitsandbytes

from transformers import AutoTokenizer, LlamaForCausalLM
import time, torch, re, numpy as np

class DetectLlama:
    def __init__(self):
        self.model = LlamaForCausalLM.from_pretrained("model_converted/llama-7b/", load_in_8bit=True, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained("model_converted/tokenizer/")

        self.max_length = 512

    def clean_text(self, text:str):
        text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        while "  " in text:
            text = text.replace("  ", " ")
        text = text.strip()
        text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" :", ":").replace(" ;", ";")
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text

    def calculate(self, text:str):
        text = self.clean_text(text)
        encodings = self.tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        # Compute log-likelihood of the sentence        
        for begin_loc in range(1, seq_len-1):
            
            tokens = encodings.input_ids[:, 0:begin_loc]

            with torch.no_grad():
                outputs = self.model(tokens)
                # Apply softmax to calculate probabilities
                probs = torch.softmax(outputs.logits.float(), dim=-1)
                
                # Get the probability of the next word
                prob = probs[0, -1, encodings.input_ids[0, begin_loc]].item()
                yield {
                    "text": self.tokenizer.decode(encodings.input_ids[0, begin_loc]),
                    "score": prob,
                }
    