# DENSE
ICASSP2025: Dynamic Embedding Causal Target Speech Extraction

Official Code for Dynamic Embedding Causal Target Speech Extraction (ICASSP 2025)

For the demos, please visit [Demo](https://wyw97.github.io/DENSE/)

## Introduction



![ar_pairs_tse_autoregressive](https://github.com/user-attachments/assets/4f74866d-2af9-4b26-bba9-bc8d3c48bb00)


## Code 

Due to the size limited, large files are not uploaded.

Most of the codes remain same as [TD-SpeakerBeam](https://github.com/butspeechfit/speakerbeam). Thanks for open source.

Some changes:

1. see egs/libri2mix_sep2vec

2. For eval.py to use real chunk-wise causal target speech extraction, DDP is used and the parameter (1531) is set fixed as the receptive length of TCN for simply.

3. For system.py, please carefully check SystemPredictedTeacherForcing and SystemPredictedParis. Some parameters should be checked before running. Sorry for the inconvenience. 

4. Open-sourced without carefully check the code style, maybe update sooner or later!
