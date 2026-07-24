# Third-party data, models, and software notices

This file is an engineering inventory, not legal advice. Versions actually used by a reported run
are recorded in `environment.json`; license terms must be rechecked before external distribution.

## Data and model assets

| Asset | Use | License/source |
|---|---|---|
| Zeroth-Korean | Public Korean ASR demonstration data | CC BY 4.0; <https://huggingface.co/datasets/kresnik/zeroth_korean> |
| OpenAI Whisper code and weights | Baseline, model comparison, LoRA | MIT; <https://github.com/openai/whisper/blob/main/LICENSE> |
| Systran Faster-Whisper converted models | Optimized inference | Model card and upstream Whisper license; <https://huggingface.co/Systran> |

CC BY 4.0 attribution for Zeroth-Korean is maintained in `DATASET_NOTICE.md`. If public data is
cropped, sampled, converted, or otherwise modified, the submission must identify that change.

## Direct software dependencies

| Package | Typical license | Project |
|---|---|---|
| faster-whisper | MIT | <https://github.com/SYSTRAN/faster-whisper> |
| CTranslate2 | MIT | <https://github.com/OpenNMT/CTranslate2> |
| huggingface-hub, transformers, datasets, accelerate, evaluate, PEFT | Apache-2.0 | <https://github.com/huggingface> |
| jiwer | Apache-2.0 | <https://github.com/jitsi/jiwer> |
| PyTorch | BSD-style | <https://github.com/pytorch/pytorch> |
| pandas | BSD-3-Clause | <https://github.com/pandas-dev/pandas> |
| psutil | BSD-3-Clause | <https://github.com/giampaolo/psutil> |
| matplotlib | Matplotlib license | <https://github.com/matplotlib/matplotlib> |
| python-docx, PyYAML, Typer | MIT | Their installed package metadata and upstream repositories |

Transitive dependencies and system libraries such as CUDA, cuDNN, FFmpeg or libsndfile may have
additional terms. Preserve the installed-environment inventory and review the complete dependency
tree for the final deployment image.
