uv run python -c "
import torch
import torch._dynamo
from transformers import AutoConfig
from model import Llama

config = AutoConfig.from_pretrained('HuggingFaceTB/SmolLM-360M')
model = Llama(model_config=config).cuda()

dummy = torch.randint(0, 1000, (2, 512)).cuda()

# Match training conditions with autocast
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    explanation = torch._dynamo.explain(model)(dummy)

print('='*60)
print('GRAPH BREAK SUMMARY')
print('='*60)
print(f'Graph count: {explanation.graph_count}')
print(f'Graph breaks: {explanation.graph_break_count}')
print(f'Break reasons:')
for reason in explanation.break_reasons:
    print(f'  - {reason}')
"