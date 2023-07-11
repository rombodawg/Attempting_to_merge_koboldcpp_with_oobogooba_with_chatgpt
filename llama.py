
class LlamaLayer:
    def __init__(self):
        self.attention_norm = None
        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None
        self.ffn_norm = None
        self.w1 = None
        self.w2 = None
        self.w3 = None

class LlamaVocab:
    def __init__(self):
        self.token_score = None

class LlamaLoadTensor:
    def __init__(self):
        self.name = ""
        self.type = None
        self.ne = []
        self.file_off = 0
        self.size = 0
        self.ggml_tensor = None
        self.data = None

class LlamaLoadTensorsMap:
    def __init__(self):
        self.tensors = []
        self.name_to_idx = {}

class LlamaSpSymbol:
    def __init__(self):
        self.prev = None
        self.next = None
        self.text = ""
        self.n = 0
