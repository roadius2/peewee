# GPU validation report

host: `trinity-prime`  python: 3.13.7  torch: 2.11.0+cu128  laya: 0.4.0.dev0  date: 2026-09-20 21:55

## english

- load: 3.8s on cuda (torch.bfloat16), 421,293,827 params, 1607 MB VRAM, NVIDIA GeForce RTX 5090

| state | intent | conf | frustration | churn | truncated |
|---|---|---|---|---|---|
| Hi, we were billed twice for March. Please refund the duplic | refund | 1.00 | 1.62 | 0.58 | False |
| The API has been returning 502s since 9am and our checkout i | technical_help | 1.00 | 1.51 | 0.08 | False |
| Je veux annuler mon forfait, votre service ne marche pas dep | refund | 0.91 | 2.27 | 0.30 | False |
| मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें। | other | 0.54 | 1.02 | 0.08 | False |
| Can I get pricing for 50 seats on the enterprise tier? | other | 0.52 | 0.66 | 0.00 | False |
| Ich kann mich seit gestern nicht einloggen, bitte helfen Sie | other | 0.55 | 1.73 | 0.16 | False |

| questions per call | p50 ms |
|---|---|
| questions_1 | 6.8 |
| questions_5 | 9.7 |
| questions_10 | 12.9 |
| questions_50 | 43.8 |
| predict_many_32x5 | 162.6 (1.02/question) |

| onnx | max abs prob diff | argmax agreement | 1q p50 ms (CPU) | 5q p50 ms (CPU) |
|---|---|---|---|---|
| fp32 | 0.0195 | 1.000 | 63.7 | 234.3 |
| int8 | 0.0530 | 1.000 | 87.1 | 280.4 |

- calibration fit on {'choice:11+': 438} examples; held-out ECE 0.344 -> 0.088, NLL 2.964 -> 2.188, accuracy 0.359
- per-locale accuracy: {"en-US": 0.622, "fr-FR": 0.444, "de-DE": 0.256, "es-ES": 0.4, "hi-IN": 0.067, "ja-JP": 0.367}
- temperatures: [2.4410483837127686, 1.0, 1.0] / {'choice:11+': 2.4410483837127686}

## multilingual

- load: 3.2s on cuda (torch.bfloat16), 321,908,995 params, 1256 MB VRAM, NVIDIA GeForce RTX 5090

| state | intent | conf | frustration | churn | truncated |
|---|---|---|---|---|---|
| Hi, we were billed twice for March. Please refund the duplic | refund | 1.00 | 2.13 | 0.10 | False |
| The API has been returning 502s since 9am and our checkout i | technical_help | 0.93 | 2.07 | 0.01 | False |
| Je veux annuler mon forfait, votre service ne marche pas dep | cancellation | 1.00 | 2.28 | 0.20 | False |
| मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें। | refund | 1.00 | 2.26 | 0.57 | False |
| Can I get pricing for 50 seats on the enterprise tier? | information | 0.92 | 1.91 | 0.01 | False |
| Ich kann mich seit gestern nicht einloggen, bitte helfen Sie | technical_help | 1.00 | 2.37 | 0.00 | False |

| questions per call | p50 ms |
|---|---|
| questions_1 | 5.6 |
| questions_5 | 6.3 |
| questions_10 | 8.4 |
| questions_50 | 22.6 |
| predict_many_32x5 | 67.3 (0.42/question) |

| onnx | max abs prob diff | argmax agreement | 1q p50 ms (CPU) | 5q p50 ms (CPU) |
|---|---|---|---|---|
| fp32 | 0.0099 | 1.000 | 26.0 | 89.9 |
| int8 | 0.0384 | 1.000 | 32.8 | 95.7 |

- calibration fit on {'choice:11+': 438} examples; held-out ECE 0.373 -> 0.130, NLL 2.476 -> 1.803, accuracy 0.435
- per-locale accuracy: {"en-US": 0.467, "fr-FR": 0.4, "de-DE": 0.422, "es-ES": 0.4, "hi-IN": 0.433, "ja-JP": 0.489}
- temperatures: [2.0730831623077393, 1.0, 1.0] / {'choice:11+': 2.0730831623077393}

- IMDB review tokens p50 211, p90 476, max 1272; trained max_len 1024

| setting | accuracy | ECE | mean conf | truncated |
|---|---|---|---|---|
| max_len=512,right | 0.900 | 0.038 | 0.927 | 9% |
| max_len=512,left | 0.903 | 0.036 | 0.927 | 9% |
| max_len=1024,right | 0.903 | 0.034 | 0.929 | 1% |
| max_len=1024,left | 0.903 | 0.034 | 0.929 | 1% |
| max_len=2048,right | 0.903 | 0.034 | 0.929 | 0% |
| max_len=2048,left | 0.903 | 0.034 | 0.929 | 0% |
| max_len=4096,right | 0.903 | 0.034 | 0.929 | 0% |
| max_len=4096,left | 0.903 | 0.034 | 0.929 | 0% |

## typed-decisions

- load: 2.6s on cuda (torch.bfloat16), 421,293,827 params, 1617 MB VRAM, NVIDIA GeForce RTX 5090

| state | intent | conf | frustration | churn | truncated |
|---|---|---|---|---|---|
| Hi, we were billed twice for March. Please refund the duplic | refund | 0.76 | 2.08 | 0.61 | False |
| The API has been returning 502s since 9am and our checkout i | technical_help | 0.95 | 1.91 | 0.38 | False |
| Je veux annuler mon forfait, votre service ne marche pas dep | cancellation | 0.54 | 2.38 | 0.67 | False |
| मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें। | other | 0.54 | 0.98 | 0.17 | False |
| Can I get pricing for 50 seats on the enterprise tier? | information | 0.50 | 0.75 | 0.03 | False |
| Ich kann mich seit gestern nicht einloggen, bitte helfen Sie | technical_help | 0.74 | 1.92 | 0.34 | False |

| questions per call | p50 ms |
|---|---|
| questions_1 | 6.9 |
| questions_5 | 9.9 |
| questions_10 | 13.1 |
| questions_50 | 44.2 |
| predict_many_32x5 | 164.5 (1.03/question) |

| onnx | max abs prob diff | argmax agreement | 1q p50 ms (CPU) | 5q p50 ms (CPU) |
|---|---|---|---|---|
| fp32 | 0.0082 | 1.000 | 63.8 | 238.3 |
| int8 | 0.0192 | 0.967 | 86.9 | 307.4 |

- IMDB review tokens p50 215, p90 485, max 1291; trained max_len 1024

| setting | accuracy | ECE | mean conf | truncated |
|---|---|---|---|---|
| max_len=512,right | 0.930 | 0.048 | 0.891 | 10% |
| max_len=512,left | 0.930 | 0.040 | 0.892 | 10% |
| max_len=1024,right | 0.930 | 0.046 | 0.893 | 1% |
| max_len=1024,left | 0.930 | 0.043 | 0.893 | 1% |
| max_len=2048,right | 0.930 | 0.046 | 0.894 | 0% |
| max_len=2048,left | 0.930 | 0.046 | 0.894 | 0% |
| max_len=4096,right | 0.930 | 0.046 | 0.894 | 0% |
| max_len=4096,left | 0.930 | 0.046 | 0.894 | 0% |
