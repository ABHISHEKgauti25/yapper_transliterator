# Test Benchmark Report

## Systems

| system | strategy |
|---|---|
| **yapper_map** | Dictionary/map only |
| **yapper_map_fb_indicxlit** | Map + IndicXlit fallback |
| **yapper_map_fb_lipi** | Map + Lipi fallback |
| **lipi (ours)** | Lipi model only |
| **indicxlit** | AI4Bharat IndicXlit |
| **indic-transliteration (ITRANS)** | ITRANS rule-based baseline |

## Dakshina Sentences — Test

Sentences: **5,000**  
Source: `hi.romanized.rejoined.test.{native,roman}.txt`

| system | coverage | word_match | WER | CER (covered) | CER (all) | NEWS F | chrF++ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **yapper_map** | 99.89% | 77.93% | 19.76% | 4.67% | 4.81% | 93.48 | 75.19 |
| **yapper_map_fb_indicxlit** | 100.00% | 78.82% | 19.73% | 4.71% | 4.71% | 93.57 | 75.27 |
| **yapper_map_fb_lipi** | 100.00% | 78.83% | 19.71% | 4.70% | 4.70% | 93.58 | 75.28 |
| **lipi (ours)** | 100.00% | 65.40% | 33.36% | 8.11% | 8.11% | 91.49 | 67.20 |
| **indicxlit** | 100.00% | 31.83% | 67.49% | 16.99% | 16.99% | 88.06 | 52.12 |
| **indic-transliteration (ITRANS)** | 99.84% | 30.08% | 63.07% | 16.73% | 16.73% | 86.61 | 48.81 |

## Dakshina Words — Test

Words: **2,500** · Average variants per word: **1.80**  
Source: `hi.translit.sampled.test.tsv`

| system | coverage | in_attested | exact_top1 | CER (covered) | NEWS F (max-ref) |
|---|---:|---:|---:|---:|---:|
| **yapper_map** | 100.00% | 98.32% | 85.36% | 0.32% | 99.81 |
| **yapper_map_fb_indicxlit** | 100.00% | 98.32% | 85.36% | 0.32% | 99.81 |
| **yapper_map_fb_lipi** | 100.00% | 98.32% | 85.36% | 0.32% | 99.81 |
| **lipi (ours)** | 100.00% | 78.04% | 62.12% | 4.47% | 96.95 |
| **indicxlit** | 100.00% | 73.12% | 56.48% | 5.81% | 96.10 |
| **indic-transliteration (ITRANS)** | 100.00% | 26.68% | 19.96% | 22.41% | 85.95 |

## Aksharantar — Test

Word pairs: **10,112**  
Source: `hin_test.json`

| system | coverage | exact_match (covered) | CER (covered) | NEWS F |
|---|---:|---:|---:|---:|
| **yapper_map** | 53.59% | 57.98% | 9.89% | 93.05 |
| **yapper_map_fb_indicxlit** | 100.00% | 47.30% | 12.95% | 91.09 |
| **yapper_map_fb_lipi** | 100.00% | 46.29% | 13.06% | 91.00 |
| **lipi (ours)** | 100.00% | 39.97% | 14.23% | 89.94 |
| **indicxlit** | 100.00% | 39.73% | 14.56% | 89.69 |
| **indic-transliteration (ITRANS)** | 100.00% | 11.82% | 28.78% | 80.21 |

## IndoNLP 2025 Hindi — Set 1

Sentences: **9,984**  
Source: `Hindi Test Set 1.txt`

| system | coverage | word_match | WER | CER (covered) | CER (all) | NEWS F | chrF++ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **yapper_map** | 99.90% | 25.17% | 85.26% | 62.11% | 62.24% | 57.15 | 35.61 |
| **yapper_map_fb_indicxlit** | 100.00% | 25.33% | 85.29% | 62.26% | 62.26% | 57.19 | 35.64 |
| **yapper_map_fb_lipi** | 100.00% | 25.33% | 85.29% | 62.26% | 62.26% | 57.19 | 35.64 |
| **lipi (ours)** | 100.00% | 21.08% | 90.36% | 64.44% | 64.44% | 56.42 | 32.73 |
| **indicxlit** | 100.00% | 10.08% | 104.88% | 71.23% | 71.23% | 54.94 | 27.05 |
| **indic-transliteration (ITRANS)** | 99.84% | 10.23% | 102.69% | 69.31% | 69.31% | 54.84 | 26.01 |

## IndoNLP 2025 Hindi — Set 2

Sentences: **4,991**  
Source: `Hindi Test Set 2.txt`

| system | coverage | word_match | WER | CER (covered) | CER (all) | NEWS F | chrF++ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **yapper_map** | 99.87% | 52.47% | 47.69% | 23.94% | 24.01% | 82.54 | 54.92 |
| **yapper_map_fb_indicxlit** | 100.00% | 52.93% | 47.69% | 23.93% | 23.93% | 82.65 | 54.98 |
| **yapper_map_fb_lipi** | 100.00% | 52.94% | 47.68% | 23.92% | 23.92% | 82.66 | 54.98 |
| **lipi (ours)** | 100.00% | 45.67% | 55.30% | 27.12% | 27.12% | 80.98 | 50.30 |
| **indicxlit** | 100.00% | 18.60% | 84.25% | 36.90% | 36.90% | 77.52 | 38.28 |
| **indic-transliteration (ITRANS)** | 99.85% | 22.06% | 77.29% | 35.52% | 35.52% | 76.72 | 36.60 |

## FIRE 2013 Transliterated Search

Hindi tokens: **2,420**  
Source: `HindiEnglish_FIRE2013_AnnotatedDev.txt`

| system | coverage | exact_match (covered) | CER (covered) | NEWS F |
|---|---:|---:|---:|---:|
| **yapper_map** | 95.17% | 78.07% | 7.33% | 95.30 |
| **yapper_map_fb_indicxlit** | 100.00% | 76.49% | 7.90% | 94.91 |
| **yapper_map_fb_lipi** | 100.00% | 76.61% | 7.72% | 95.05 |
| **lipi (ours)** | 100.00% | 73.10% | 8.24% | 94.76 |
| **indicxlit** | 100.00% | 34.75% | 21.09% | 85.15 |
| **indic-transliteration (ITRANS)** | 100.00% | 43.88% | 20.19% | 88.73 |
