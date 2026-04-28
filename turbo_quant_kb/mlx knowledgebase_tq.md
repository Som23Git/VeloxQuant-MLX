
Freedium
ko-fi
librepay
patreon
paypal
< Go to the original

Preview image
I Built a Quantization Method That Beats Standard 4-bit on a 7B Model — With Zero Training Data
Adapting Google's TurboQuant for weight compression on Apple Silicon using MLX, and what the perplexity numbers actually tell us
Manjunath Janardhan
Manjunath Janardhan

Follow
AI Advances
AI Advances

androidstudio
~7 min read
·
April 3, 2026 (Updated: April 3, 2026)
·
Free: No
Last year, I bought a 64GB M4 Max MacBook to run larger models. The FP16 weights alone take around 14GB of unified memory. The obvious fix is quantization — squeeze those 16-bit weights down to 4 or 3 bits.

MLX makes this simple. Get the MLX model from Huggingface / LM Studio, or Ollama, and you'll get a 4-bit MLX model. But when I tried 3-bit? The outputs were gibberish. And 2-bit was completely unusable — perplexity over 2000.

One way to solve this is by using Affine (uniform) quantisation, which spreads its reconstruction levels evenly across the value range. But transformer weights aren't evenly distributed — they cluster near zero with heavy tails. You end up wasting precious bits on sparse tail regions while under-representing the dense center where most values live.

Last week, I found Google's TurboQuant paper. They'd built a clever two-stage pipeline for KV cache compression. I spent some time adapting it for something different: full-weight quantization on Apple Silicon.

The results were excellent.

The idea that changed everything
Here's what TurboQuant does differently, and why it clicked for me.

First, make the problem easier. Before quantizing, multiply every weight row by a randomized Hadamard matrix. This is a fast orthogonal rotation — O(d log d) instead of O(d²) — and MLX has a native Metal-accelerated kernel for it. The rotation spreads outlier energy across all channels. What you get out the other side is weights that are approximately Gaussian.

I measured this on actual model weights. The first attention layer in SmolLM2–135M had a kurtosis of 10.99 (Gaussian is 3.0) and a 21x ratio between the largest and smallest channel magnitudes. After rotation: kurtosis dropped to 6.37, channel ratio dropped to 3.15x. That's a 42% reduction in kurtosis and an 85% improvement in channel uniformity. The rotation really does Gaussianize the weights.

None
Image by Manjunath Janardhan. Kurtosis before and after Hadamard rotation across transformer layers.
The dashed green line marks perfect Gaussian (3.0).

Then, use the optimal quantizer for Gaussians. Once you know the distribution is Gaussian, the best possible scalar quantizer is a solved problem — Lloyd and Max worked this out in the 1960s. The Lloyd-Max codebook places more reconstruction levels near zero (high density) and fewer in the tails (low density). The codebook is a mathematical constant. You compute it once for N(0,1) and use it forever. No calibration set, no fine-tuning, no data at all.

The per-element MSE advantage over uniform quantization is about 7%. That sounds small, but it compounds through 30+ transformer layers. The effective SNR improvement works out to about 9 dB at 3-bit quantization. In practice, that's the difference between coherent text and noise.

Finally, clean up what's left. After the first stage, there's still a quantization residual. TurboQuant's second stage compresses this residual to just 1 bit per element using a technique called QJL (Quantized Johnson-Lindenstrauss). You project through another randomized Hadamard, keep only the sign bits, and store one L2 norm per row. At inference, this gives you a mathematically unbiased correction to the dot product.

How much could 1-bit signs actually help? The numbers : 43–45% MSE reduction at every bit-width I tested.

None
Image by Manjunath Janardhan. QJL residual correction (Stage 2) consistently reduces quantization error by 43–45%
What the numbers say
I ran WikiText-2 perplexity evaluations across four models: Llama-3.2–1B, Llama-3.2–3B, Qwen3.5–4B, and Qwen2.5–7B. Each model had— FP16 baseline, TurboQuant at 2/3/4 bits with and without QJL, and MLX affine quantization at 2/3/4 bits. That's 10 configurations per model, 256 chunks of 512 tokens each.

The 7B results are the headline.

None
Image By Manjunath Janardhan. Full WikiText-2 perplexity results on Qwen2.5–7B (7.62B parameters). Blue rows: TurboQuant configurations. Red rows: Affine baselines.
At 7B, TurboQuant wins at every bit-width:

- 4-bit: TQ 9.14 vs Affine 9.24 — TurboQuant is lower (better)

- 3-bit: TQ 11.69 vs Affine 13.37 — a meaningful gap

- 4+QJL: TQ 8.92 — only 4% above FP16's 8.55

- 2+QJL: TQ 15.93 vs Affine 2199 — this isn't a typo, affine 2-bit is completely broken

That last number shows that Usable 2-bit quantization with no training data at all.

The scaling story
The most interesting finding wasn't any single number — it was the trend across model sizes.

None
Image By Manjunath Janardhan. WikiText-2 perplexity across four models from 1B to 7B.urboQuant's advantage grows with scale.
None
Image by Manjunath Janardhan. 4-bit quantization comparison across model scales. TurboQuant (blue) tracks FP16 (black) more closely than Affine (red), especially at 7B.
At 1B parameters, TurboQuant and affine are roughly tied. By 3B, TurboQuant starts pulling ahead. At 4B and 7B, the gap becomes decisive.

This makes theoretical sense — the Hadamard rotation's Gaussianization effect is a consequence of concentration of measure in high dimensions. Bigger models have wider layers (4096-dim for 7B vs 2048 for 1B), so the Gaussian approximation is tighter.

The 2-bit results tell an even more dramatic story.

None
Image by Manjunath janardhan. Left: TurboQuant 2+QJL perplexity drops sharply with model size. Right: Log-scale comparison shows TQ is orders of magnitude better than affine at 2-bit.
TurboQuant 2+QJL goes from PPL 410 at 1B (not great) to 91 at 3B to 21.5 at 4B to 15.9 at 7B. Meanwhile, affine 2-bit is above 700 on every model. At 7B, affine 2-bit gives you PPL 2199 — essentially random text. TurboQuant 2+QJL at 15.9 is actually usable.

The implementation issues
Building this on MLX wasn't all smooth sailing. A few issues I would like to share so that they might save someone else time.

Hadamard dimensions are picky. MLX's `mx.hadamard_transform()` only supports dimensions n = m * 2^k where m is in {1, 12, 20, 28}. That's fine for Llama models (2048, 4096 — clean powers of 2), but Qwen2.5–7B has a hidden size of 3584 = 7 x 512. My QJL code was calling `mx.hadamard_transform()` directly and crashing on 7B with a dimension error.

The fix was a blockwise approach: split into compatible blocks (3584 = 7 blocks of 512) and transform each block independently. The PolarQuant rotation code already handled this, but I'd missed it in the QJL path. Classic case of the same bug hiding in two different code paths.

Resource management matters. I initially tried running three model evaluations in parallel (3B, 4B, 7B). The 7B evaluation died from memory pressure — semaphore leak while loading a 2-bit model. Had to kill it and re-run solo after the others finished. On a 64GB M4 Max. These models are hungry.

The fused Metal kernel. I wrote a custom Metal kernel that fuses index unpacking, codebook lookup, scale multiply, and dot product into one GPU pass. Instead of materializing the full FP16 weight matrix (33MB for a 4096x4096 layer), the kernel reads ~7MB of compressed data directly. The speedup at LLM-scale dimensions: 3.8–4.3x. At smaller layers (512x512), the overhead of kernel dispatch dominates, and you only get 1.3x. The lesson: custom kernels pay off at scale, not at toy sizes.

What I got wrong (and what's still missing)
I want to be upfront about the limitations because I think they matter for anyone considering this approach.

TurboQuant models are bigger than they should be. I don't quantize embedding layers — they stay FP16. For Llama-3.2–1B with its 128K-token vocabulary, the embedding is 21% of all parameters. That means at 1B scale, affine quantization (which does quantize embeddings) actually gives you a better size-quality tradeoff. The TurboQuant advantage only shows up at 3B+ where embeddings are a smaller fraction.

I haven't compared against GPTQ or AWQ. Those are calibration-based methods that might still win at equal bit-width. But they need calibration data and take longer to run. TurboQuant's zero-data property is a genuine practical advantage — you can quantize any model in seconds without curating a calibration set.

The non-monotonic scaling at 4B to 7B. If you look closely at the 4+QJL gap to FP16, it goes +10% (1B) -> +6% (3B) -> +1% (4B) -> +4% (7B). That jump from +1% to +4% bothered me until I realized the 4B model is Qwen3.5 and the 7B is Qwen2.5 — different architectures. Within each model family, the trend is consistent. Still, more data points across the same-family models would strengthen the scaling story.

Next Steps
Three things I want to do:

1. Quantize embeddings — this should close the size gap with affine and make TurboQuant competitive even at 1B

2. Test on 13B-70B — the theory predicts even stronger Gaussianization at higher dimensions, so the advantage should grow

3. Head-to-head with GPTQ/AWQ — the benchmark which I would like to see.

An arXiv preprint is coming soon with the full technical details, proofs, and all benchmark data. The code will be open-sourced as soon as it is published.

The bottom line
If you're quantizing models for Apple Silicon and you care about quality below 4 bits, the standard affine approach is leaving performance on the table. Rotating weights to Gaussian and using the optimal Gaussian quantizer is a small conceptual change that yields measurably better results — and it requires zero training data.

The effect is strongest exactly where you want it: on the bigger models that need compression the most.

Support
If you found this article informative and valuable, I'd greatly appreciate your support:

"Give it a few claps 👏 on Medium to help others discover this content (did you know you can clap up to 50 times?). Your claps will help spread the knowledge to more readers."

Share it with your network of AI enthusiasts and professionals.
Subscribe to my YouTube channel for AI videos explained in simple English: https://www.youtube.com/@AIBroEnglish
Connect with me on LinkedIn: https://www.linkedin.com/in/manjunath-janardhan-54a5537/
Thanks For Reading
#turbo-quant
#quantization
#apple-silicon
#llm
< Go to the original

