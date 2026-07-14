# Development and Validation Benchmark Report

## Systems

| system | strategy |
|---|---|
| **yapper_map** | Dictionary/map only |
| **yapper_map_fb_indicxlit** | Map + IndicXlit fallback |
| **yapper_map_fb_lipi** | Map + Lipi fallback |
| **lipi (ours)** | Lipi model only |
| **indicxlit** | AI4Bharat IndicXlit |
| **indic-transliteration (ITRANS)** | ITRANS rule-based baseline |

## Dakshina Sentences — Dev

Sentences: **5,000**  
Source: `hi.romanized.rejoined.dev.{native,roman}.txt`

| system | coverage | word_match | WER | CER (covered) | CER (all) | NEWS F | chrF++ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **yapper_map** | 99.91% | 78.29% | 19.51% | 4.60% | 4.72% | 93.73 | 75.69 |
| **yapper_map_fb_indicxlit** | 100.00% | 79.10% | 19.48% | 4.64% | 4.64% | 93.82 | 75.75 |
| **yapper_map_fb_lipi** | 100.00% | 79.11% | 19.47% | 4.63% | 4.63% | 93.82 | 75.75 |
| **lipi (ours)** | 100.00% | 65.75% | 32.92% | 7.93% | 7.93% | 91.76 | 67.87 |
| **indicxlit** | 100.00% | 31.68% | 67.55% | 16.93% | 16.93% | 88.20 | 52.40 |
| **indic-transliteration (ITRANS)** | 99.84% | 29.50% | 63.34% | 16.79% | 16.79% | 86.75 | 48.66 |

## Dakshina Words — Dev

Words: **2,500** · Average variants per word: **1.74**  
Source: `hi.translit.sampled.dev.tsv`

| system | coverage | in_attested | exact_top1 | CER (covered) | NEWS F (max-ref) |
|---|---:|---:|---:|---:|---:|
| **yapper_map** | 100.00% | 98.84% | 85.72% | 0.25% | 99.85 |
| **yapper_map_fb_indicxlit** | 100.00% | 98.84% | 85.72% | 0.25% | 99.85 |
| **yapper_map_fb_lipi** | 100.00% | 98.84% | 85.72% | 0.25% | 99.85 |
| **lipi (ours)** | 100.00% | 79.00% | 64.20% | 4.36% | 97.10 |
| **indicxlit** | 100.00% | 72.68% | 57.16% | 6.08% | 95.81 |
| **indic-transliteration (ITRANS)** | 100.00% | 24.24% | 17.96% | 23.28% | 85.41 |

## Aksharantar — Valid

Word pairs: **6,357**  
Source: `hin_valid.json`

| system | coverage | exact_match (covered) | CER (covered) | NEWS F |
|---|---:|---:|---:|---:|
| **yapper_map** | 69.91% | 56.37% | 10.30% | 92.83 |
| **yapper_map_fb_indicxlit** | 100.00% | 62.92% | 8.04% | 94.08 |
| **yapper_map_fb_lipi** | 100.00% | 60.70% | 8.45% | 93.83 |
| **lipi (ours)** | 100.00% | 53.08% | 9.89% | 92.68 |
| **indicxlit** | 100.00% | 52.82% | 10.47% | 92.12 |
| **indic-transliteration (ITRANS)** | 100.00% | 14.41% | 26.80% | 81.89 |
