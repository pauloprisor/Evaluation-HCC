import sys
import os
import time
import json
import torch
import torch.nn as nn

# Monkey patch .cuda() to avoid crashes on macOS and map to MPS
original_tensor_cuda = torch.Tensor.cuda
def patched_tensor_cuda(self, *args, **kwargs):
    if torch.backends.mps.is_available():
        return self.to('mps', *args, **kwargs)
    return self.to('cpu', *args, **kwargs)
torch.Tensor.cuda = patched_tensor_cuda

original_module_cuda = nn.Module.cuda
def patched_module_cuda(self, *args, **kwargs):
    if torch.backends.mps.is_available():
        return self.to('mps', *args, **kwargs)
    return self.to('cpu', *args, **kwargs)
nn.Module.cuda = patched_module_cuda

from flask import Flask, request, jsonify

# Add cpc_repo to path so imports work correctly
cpc_repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'cpc_repo'))
sys.path.insert(0, cpc_repo_path)

from prompt_compressor import PromptCompressorCPC
from model.common import ModelType

app = Flask(__name__)
compressor = None

print("[CPC Server] Initializing CPC Compressor...")
# Llama model config
compressor = PromptCompressorCPC(
    model_type=ModelType.LLAMA,
    use_question_as_suffix=False,
    use_openai_tokenizer_to_measure_length=True
)
print("[CPC Server] Compressor initialized and ready!")

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok"})

@app.route('/compress', methods=['POST'])
def compress():
    data = request.json
    context = data.get('context', '')
    question = data.get('question', '')
    token_budget = data.get('token_budget', 1000)

    try:
        compressed_text = compressor.compress(
            context=context,
            question=question,
            compression_target_tokens=token_budget,
        )
        return jsonify({"compressed_text": compressed_text})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)
