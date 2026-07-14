---
language:
- hi
license: other
library_name: pytorch
tags:
- yapper-transliterator
- lipi
- transliteration
- romanization
- hindi
- devanagari
- ctc
- bigru
- edge-ai
- on-device
metrics:
- accuracy
- cer
- wer
- chrf
---

# Lipi

**Lipi is the compact Hindi Devanagari-to-Roman model released as part of Yapper Transliterator.**

Lipi converts Hindi written in Devanagari into natural Roman Hindi using a non-autoregressive architecture designed for CPU and edge inference.

```text
मैंने कहा कि भारत महान है
→ maine kaha ki bharat mahan hai
```

The full Yapper Transliterator system combines Lipi with `yapper_map`, a curated 1.17M-entry dictionary. This card describes the **standalone Lipi model** and separately reports full-system results where relevant.

## Model details

| Property | Value |
|---|---|
| Model name | **Lipi** |
| Project | **Yapper Transliterator** |
| Task | Hindi Devanagari → Roman transliteration |
| Parameters | **544,653** |
| Architecture | Akshara/codepoint encoder → slot expansion → 2-layer BiGRU → CTC |
| Decoding | Greedy CTC; optional prefix beam search |
| Source vocabulary | 129 symbols |
| Target vocabulary | 77 symbols |
| Character embedding size | 32 |
| Model dimension | 96 |
| GRU hidden size | 128 per direction |
| GRU layers | 2 |
| Dropout | 0.1 |
| Maximum codepoints per cluster | 16 |
| Output slots per akshara | 6 |
| Framework | PyTorch |

Lipi keeps Devanagari orthographic clusters (**aksharas**) intact while composing each cluster from its Unicode codepoints. This gives unseen conjuncts a meaningful representation without requiring every cluster to appear as a dedicated vocabulary item.

## Intended use

- Hindi Devanagari-to-Roman transliteration for messaging, dictation, search, and Hinglish interfaces.
- Lightweight fallback behind a deterministic transliteration dictionary.
- CPU-first, local, and edge inference.
- Research on compact non-autoregressive transliteration.

## Out-of-scope use

- Roman-to-Devanagari conversion.
- Languages other than Hindi without additional training and validation.
- ISO, IAST, ITRANS, or other formal romanization standards.
- Translation, rewriting, grammatical correction, or speech recognition.
- Treating one Roman spelling as the only linguistically valid spelling.

## Usage

Lipi uses the Yapper Transliterator runtime rather than a standard Transformers pipeline.

```python
from yapper_transliterator import Transliterator

model = Transliterator(
    backend="lipi",
    model_dir="/path/to/downloaded/lipi",
)

print(model.transliterate_text("नमस्ते दुनिया"))
# namaste duniya
```

Recommended full Yapper Transliterator system:

```python
from yapper_transliterator import Transliterator

model = Transliterator(
    backend="map_lipi",
    model_dir="/path/to/downloaded/lipi",
    map_path="/path/to/yapper_map.json",
)
```

Command line:

```bash
python transliterate.py \
  --text "मैंने कहा" \
  --backend lipi \
  --model-dir /path/to/downloaded/lipi
```

## Training data

The released Lipi checkpoint was trained on **1,149,086 prepared Hindi Devanagari–Roman word pairs** produced by the Yapper Transliterator data pipeline. The pipeline draws from curated and third-party transliteration resources, including Dakshina, Xlit-Crowd, L3Cube data, and Aksharantar-derived entries.

The exact licenses and redistribution conditions differ across sources. The Lipi checkpoint should not automatically be assumed to inherit the Yapper Transliterator code repository's Apache-2.0 license.

## Evaluation

### Lipi standalone

| Benchmark | Split | Metric | Result |
|---|---|---|---:|
| Dakshina sentences | test, 5,000 sentences | Word match | 65.40% |
|  |  | CER | 8.11% |
|  |  | NEWS F | 91.49 |
| Dakshina words | test, 2,500 words | In-attested | 78.04% |
|  |  | Nearest-reference CER | 4.47% |
| Aksharantar | test, 10,112 words | Exact match | 39.97% |
|  |  | CER | 14.23% |
| FIRE 2013 | dev, 2,420 Hindi tokens | Exact match | 73.10% |
|  |  | CER | 8.24% |
| IndoNLP 2025 | Set 2, 4,991 sentences | Word match | 45.67% |
|  |  | CER | 27.12% |

### Full Yapper Transliterator system: `yapper_map` + Lipi

| Benchmark | Main result |
|---|---:|
| Dakshina sentences test | 78.83% word match; 4.70% CER |
| Dakshina words test | 98.32% in-attested |
| Aksharantar test | 46.29% exact match |
| FIRE 2013 dev | 76.61% exact match; 7.72% CER |
| IndoNLP 2025 Set 2 | 52.94% word match; 23.92% CER |

Results are not directly interchangeable across datasets. Roman Hindi permits multiple valid spellings, several benchmarks use one reference, and IndoNLP is evaluated by reversing its original Roman-to-Devanagari direction.

## Limitations

- Multiple Roman spellings may be valid, such as `nahi` and `nahin`.
- Schwa deletion is context-dependent and remains difficult.
- Proper names, borrowings, regional forms, and rare words may be romanized unexpectedly.
- Output style reflects the distributions and annotator preferences in the training sources.
- The highest-quality Yapper Transliterator configuration uses `yapper_map` before invoking Lipi.

## Ethical and deployment considerations

Transliteration can affect names and identity-bearing text. Applications should allow users to review and edit outputs and should not treat a generated spelling as authoritative. Validate Lipi on the target domain before identity-sensitive or high-impact use.

## License

The Yapper Transliterator source code is Apache-2.0. This model card uses `license: other` because the Lipi training pipeline draws from multiple third-party datasets with separate terms. Verify all applicable licenses before redistributing the checkpoint or deploying it commercially.

## Citation

```bibtex
@software{lipi_yapper_transliterator_2026,
  title   = {Lipi: A Compact Hindi Devanagari-to-Roman Transliteration Model},
  author  = {Abhishek Gautam},
  year    = {2026},
  note    = {Released as part of Yapper Transliterator},
  url     = {https://github.com/ABHISHEKgauti25/yapper_transliterator}
}
```
