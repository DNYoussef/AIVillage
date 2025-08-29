# Mobile Readiness Report

## Resource Profiling
- **Generic 2GB Budget Phone:** CPU 60.0% | Peak Memory 501.0MB
- **Xiaomi Redmi Note 10 (4GB):** CPU 0.0% | Peak Memory 501.0MB

## Compression Benchmarks
| Model  | Precision | Original KB | Compressed KB | Ratio |
|-------|-----------|-------------|---------------|-------|
| tiny  | float32   | 38.0        | 12.9          | 2.95x |
| small | float32   | 922.5       | 235.8         | 3.91x |
| medium| float32   | 3603.7      | 909.8         | 3.96x |
| large | float32   | 14874.6     | 3732.9        | 3.98x |
| tiny  | float16   | 20.7        | 12.8          | 1.62x |
| small | float16   | 463.2       | 235.1         | 1.97x |
| medium| float16   | 1804.2      | 907.3         | 1.99x |
| large | float16   | 7440.1      | 3727.4        | 2.00x |
