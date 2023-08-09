# Installation
## Environment and Submodule Setup (AGAVE)
```
module activate llm_score
module load gcc/9.1.0
```
## GRiT Model Downloading
```
cd LLMScore/models
wget https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth && cd ..
```
