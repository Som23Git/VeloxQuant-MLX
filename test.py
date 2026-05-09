import mlx_lm
from mlx_kv_quant.weight import quantize_model, compression_report

model, tokenizer = mlx_lm.load('mlx-community/SmolLM2-135M-Instruct')
print("Quantizing model weights with TurboQuant 4-bit...")
model = quantize_model(model, bits=4, use_hadamard=True)

report = compression_report(model)
print("\n=== Compression Report ===")
print(f"Layers quantized : {report['n_layers']}")
print(f"Original size    : {report['total_fp16_mb']:.0f} MB (fp16)")
print(f"Compressed size  : {report['total_compressed_mb']:.0f} MB")
print(f"Compression ratio: {report['compression_ratio']:.1f}x")

print("\n=== Generation Test ===")
response = mlx_lm.generate(
    model, tokenizer,
    prompt="What is the capital of France?",
    max_tokens=60,
    verbose=True,
)
