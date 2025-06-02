import torch
import torch.nn as nn
import importlib.util

def load_model():
    spec = importlib.util.spec_from_file_location("model", "main.py")
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return model_module.Net()

model = load_model()
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

assert total_params < 20000, f"❌ Too many parameters: {total_params}"
print(f"✅ Param check passed: {total_params} params")

has_bn = any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
has_do = any(isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d) for m in model.modules())
has_gap_or_fc = any(isinstance(m, nn.AdaptiveAvgPool2d) or isinstance(m, nn.Linear) for m in model.modules())

assert has_bn, "❌ BatchNorm not found"
assert has_do, "❌ Dropout not found"
assert has_gap_or_fc, "❌ GAP or FC layer not found"

print("✅ BatchNorm found")
print("✅ Dropout found")
print("✅ GAP/FC layer found")
