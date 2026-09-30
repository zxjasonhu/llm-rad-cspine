| Model Variant | Method | Macro-F1 | Macro Precision | Macro Recall | Report Level Sensitivity | Report Level Specificity |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma-3 | Zero-shot (Direct) | 0.669 | 0.543 | 0.932 | 0.968 | 0.934 |
|  | Zero-shot (Impression-guided) | 0.730 | 0.616 | 0.938 | 0.968 | 0.959 |
|  | LoRA (Standard) | 0.885 ± 0.022 | 0.883 ± 0.062 | 0.899 ± 0.028 | 0.937 ± 0.025 | 0.987 ± 0.010 |
|  | LoRA (Impression-regularized) | 0.909 ± 0.004 | 0.929 ± 0.014 | 0.893 ± 0.010 | 0.943 ± 0.014 | 0.995 ± 0.002 |
| Llama-3.2 | Zero-shot (Direct) | 0.791 | 0.747 | 0.856 | 0.889 | 0.983 |
|  | Zero-shot (Impression-guided) | 0.797 | 0.776 | 0.836 | 0.905 | 0.982 |
|  | LoRA (Standard) | 0.890 ± 0.004 | 0.918 ± 0.024 | 0.867 ± 0.021 | 0.918 ± 0.012 | 0.993 ± 0.003 |
|  | LoRA (Impression-regularized) | 0.902 ± 0.004 | 0.937 ± 0.022 | 0.874 ± 0.020 | 0.924 ± 0.018 | 0.995 ± 0.003 |
| Phi-3.5 | Zero-shot (Direct) | 0.859 | 0.885 | 0.845 | 0.898 | 0.992 |
|  | Zero-shot (Impression-guided) | 0.763 | 0.938 | 0.668 | 0.760 | 0.993 |
|  | LoRA (Standard) | 0.877 ± 0.008 | 0.938 ± 0.011 | 0.827 ± 0.018 | 0.880 ± 0.011 | 0.996 ± 0.000 |
|  | LoRA (Impression-regularized) | 0.880 ± 0.003 | 0.919 ± 0.004 | 0.848 ± 0.006 | 0.897 ± 0.004 | 0.995 ± 0.000 |
