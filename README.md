# VALL-E Trainer
Simple VALL-E training code (sft and dpo). 

## Usage

### Obtain Pretrained Model and Text Tokens
Get pretrained VALL-E TTS model (ko)
```
wget https://huggingface.co/LearnItAnyway/vall-e_korean/resolve/main/vall-e_ko_v1.pt
wget https://huggingface.co/LearnItAnyway/vall-e_korean/resolve/main/unique_text_tokens_v1.k2symbols
```
The VALL-E Korean model above is pretrained using [lifeiteng/vall-e](https://github.com/lifeiteng/vall-e).
You can use other pretrained model but some modification for the text tokenizer is required.

### Train with Supervised Fine Tuning (SFT)

```
python sft_trainer.py --model_path <path-to-model> --data_dir <path-to-data> --epochs <num-epochs> --lr <learning-rate>
```

For the training data, you should put the pairs of audio-text files in the `data_dir` folder.

### Train with Direct Preference Optimization (DPO)

```
python sft_trainer.py --model_path <path-to-model> --data_dir <path-to-data> --epochs <num-epochs> --lr <learning-rate>
```

[DPO](https://arxiv.org/pdf/2305.18290.pdf) is RLHF algorithm.
For the training, preference data is required, consisting of `audio_original`, `audio_positive`, `audio_negative`, `text_original`, `text_positive`, `text_negative`.
Here, the advantage of the RLHF is that you can make `audio_positive` and `audio_negative` as many as you want using the pretrained or fine-tuned VALL-E model, where more preference dataset will improve your model.

### Running the Web UI
We provide the web UI to run the model

```
python webui.py
```

### Requirements
#### Python
gradio==3.50.2
encodec
vocos
phonemizer
torchmetrics
espeak-ng


### References

- https://github.com/lifeiteng/vall-e
- https://arxiv.org/abs/2305.18290
- https://huggingface.co/LearnItAnyway/vall-e_korean