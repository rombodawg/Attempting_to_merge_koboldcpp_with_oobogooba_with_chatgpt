
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
        # Add more attributes as needed

class LlamaVocab:
    def __init__(self):
        self.id = None
        self.token = None
        self.score = None
        # Add more attributes as needed

class LlamaLoadTensor:
    def __init__(self):
        self.name = ""
        self.type = None
        self.ne = []
        self.file_off = 0
        self.size = 0
        self.ggml_tensor = None
        self.data = None
        # Add more attributes as needed

class LlamaLoadTensorsMap:
    def __init__(self):
        self.tensors = []
        self.name_to_idx = {}
        # Add more attributes as needed

class LlamaSpSymbol:
    def __init__(self):
        self.prev = None
        self.next = None
        self.text = ""
        self.n = 0
        # Add more attributes as needed
