import argparse
import os
import time
import torch, torchaudio
import random
import tqdm
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np

from encodec import EncodecModel
from encodec.utils import convert_audio

from models.utils import get_text_token_collater, AudioTokenizer, TextTokenizer, AttributeDict, EarlyStopping
from models import get_model

from vocos import Vocos

parser = argparse.ArgumentParser(description='Audio-Text Model Training')
parser.add_argument('--model_path', type=str, default='vall-e_ko_v1.pt')
parser.add_argument('--model_ref_path', type=str, default=None)

parser.add_argument('--token_path', type=str, default='unique_text_tokens_v1.k2symbols')
parser.add_argument('--output_path', type=str, default='tmp.pt')

parser.add_argument('--data_dir', type=str, default='data_pref', help='Directory containing audio files')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
parser.add_argument('--seed', type=int, default=42, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--validation_split', type=float, default=0.2, help='Validation split ratio')
args = parser.parse_args()

seed = args.seed  # You can choose any number as a seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", 0)

language = 'en'
if args.model_path.find('ko') >=0:
    language = 'ko'
text_tokenizer = TextTokenizer(language=language)

checkpoint = torch.load(args.model_path, map_location='cpu')
if args.model_ref_path is None:
    args.model_ref_path = args.model_path
checkpoint = torch.load(args.model_path, map_location='cpu')
checkpoint_ref = torch.load(args.model_ref_path, map_location='cpu')

model = get_model(AttributeDict(checkpoint))
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint["model"], strict=True
)
assert not missing_keys

model_ref = get_model(AttributeDict(checkpoint_ref))
missing_keys, unexpected_keys = model_ref.load_state_dict(
    checkpoint_ref["model"], strict=True
)

text_collater = get_text_token_collater(args.token_path)
audio_tokenizer = AudioTokenizer(device)
assert not missing_keys

model.to(device)
model_ref.to(device)

text_collater = get_text_token_collater(args.token_path)
audio_tokenizer = AudioTokenizer(device)


class WavTextDataset(Dataset):
    def __init__(self, data_dir):
        self.aa = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('ori.wav')]
        self.bb = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('pos.wav')]
        self.cc = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('neg.wav')]
        self.dd = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('text_new.txt')]
        self.ee = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('text_ori.txt')]

    def __len__(self):
        return len(self.aa)

    def get_wav_pr(self, audio_path):
        wav_pr, sr = torchaudio.load(audio_path)
        wav_pr = convert_audio(wav_pr, sr, audio_tokenizer.sample_rate, audio_tokenizer.channels)
        return wav_pr

    def get_text(self, text_path):
        with open(text_path, 'r') as f:
            text_prompt = f.read().strip()
        return text_prompt

    def __getitem__(self, idx):

        text_prompt_new = self.get_text(self.dd[idx])
        text_prompt_ori = self.get_text(self.ee[idx])

        wav_pr_ori = self.get_wav_pr(self.aa[idx])
        wav_pr_pos = self.get_wav_pr(self.bb[idx])
        wav_pr_neg = self.get_wav_pr(self.cc[idx])

        return text_prompt_ori, text_prompt_new, wav_pr_ori, wav_pr_pos, wav_pr_neg

def get_input(text_prompt_ori, text_prompt_new, wav_pr_ori, wav_pr_pos, wav_pr_neg):
    text_prompt = f'{text_prompt_ori} {text_prompt_new}'
    audio_prompts_pos = audio_tokenizer.encode(torch.concat([wav_pr_ori, wav_pr_pos], dim=2)
                            )[0][0].transpose(2, 1).to(device)
    audio_prompts_neg = audio_tokenizer.encode(torch.concat([wav_pr_ori, wav_pr_neg], dim=2)
                            )[0][0].transpose(2, 1).to(device)
    text_tokens, text_tokens_lens = text_collater([text_tokenizer(text_prompt)])
    return text_tokens, text_tokens_lens, audio_prompts_pos, audio_prompts_neg


# Split dataset into train and validation sets
def split_dataset(dataset, validation_split=0.2):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


dataset = WavTextDataset(args.data_dir)
train_dataset, valid_dataset = split_dataset(dataset, validation_split=args.validation_split)

# Create DataLoaders for train and validation
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)
#valid_loader = train_loader

def get_dpo_loss(mode, model_ref, batch, nar_, stage, beta=0.1):
    text_prompt_ori, text_prompt_new, wav_pr_ori, wav_pr_pos, wav_pr_neg = batch
    text_tokens, _, audio_pos, audio_neg = get_input(*batch)
    text_tokens_lens = torch.LongTensor([text_tokens.shape[1]]).to(device)
    audio_pos_lens = torch.LongTensor([audio_pos.shape[1]]).to(device)
    audio_neg_lens = torch.LongTensor([audio_neg.shape[1]]).to(device)
    # Do something with the data
    assert text_tokens.ndim == 2 and  audio_pos.ndim == 3
    with torch.no_grad():
        predicts, loss, metrics, ref_logits_pos = model_ref(
            x=text_tokens.to(device),
            x_lens=text_tokens_lens,
            y=audio_pos,
            y_lens=audio_pos_lens,
            train_stage=stage,
            nar_=nar_,
            return_logits=True
        )

        predicts, loss, metrics, ref_logits_neg = model_ref(
            x=text_tokens.to(device),
            x_lens=text_tokens_lens,
            y=audio_neg,
            y_lens=audio_neg_lens,
            train_stage=stage,
            nar_=nar_,
            return_logits=True
        )
    with torch.set_grad_enabled(True):
        predicts, loss, metrics, logits_pos = model(
            x=text_tokens.to(device),
            x_lens=text_tokens_lens,
            y=audio_pos,
            y_lens=audio_pos_lens,
            train_stage=stage,
            nar_=nar_,
            return_logits=True
        )

        predicts, loss, metrics, logits_neg = model(
            x=text_tokens.to(device),
            x_lens=text_tokens_lens,
            y=audio_neg,
            y_lens=audio_neg_lens,
            train_stage=stage,
            nar_=nar_,
            return_logits=True
        )

        pos_logp_sum = torch.gather(logits_pos[0].log_softmax(-1), dim=0, index=audio_pos[..., -logits_pos.shape[-1]:, nar_]).sum()
        neg_logp_sum = torch.gather(logits_neg[0].log_softmax(-1), dim=0, index=audio_neg[..., -logits_neg.shape[-1]:, nar_]).sum()
        ref_pos_logp_sum = torch.gather(ref_logits_pos[0].log_softmax(-1), dim=0, index=audio_pos[..., -ref_logits_pos.shape[-1]:, nar_]).sum()
        ref_neg_logp_sum = torch.gather(ref_logits_neg[0].log_softmax(-1), dim=0, index=audio_neg[..., -ref_logits_neg.shape[-1]:, nar_]).sum()
        pos_logratio = pos_logp_sum/(ref_pos_logp_sum+1e-6)
        neg_logratio = neg_logp_sum/(ref_neg_logp_sum+1e-6)
        # https://arxiv.org/pdf/2305.18290v2.pdf
        dpo_loss = -torch.nn.functional.logsigmoid((pos_logratio-neg_logratio)*beta)
    return dpo_loss

def dpo_train_loop(model, model_ref, loader, valid_loader, optimizer, device, num_epochs=10, stage=0):
    # Send model to device
    # Loop over epochs
    early_stopping = EarlyStopping(patience=3)
    model = model.to(device)
    model_ref = model_ref.to(device)
    if stage==2: nars = np.arange(1, 8)
    else: nars = [0]
    for epoch in range(num_epochs):
        loss_sum = 0
        optimizer.zero_grad()
        model.train()
        model_ref.eval()
        for _, batch in enumerate(train_loader):
        #for _, batch in enumerate(tqdm.tqdm(train_loader)):
            for nar_ in nars :

                loss = get_dpo_loss(model, model_ref, batch, nar_, stage)

                loss.backward()
                loss_sum += loss.item()
        optimizer.step()

        loss_sum_ = 0
        model.eval()
        for _, batch in enumerate(valid_loader):
            for nar_ in nars :
                loss = get_dpo_loss(model, model_ref, batch, nar_, stage)
                loss_sum_ += loss.item()

        loss_sum /= (len(train_loader) * len(nars)+1e-6)
        loss_sum_ /= (len(valid_loader) * len(nars)+1e-6)
        print("Train Loss : ", loss_sum, "Valid Loss : ", loss_sum_)
        early_stopping(loss_sum_)
        if early_stopping.early_stop:
            break

    return model

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

model = dpo_train_loop(model, model_ref, train_loader, valid_loader, optimizer, device=device, num_epochs=args.epochs, stage=1)
print('Stage 1 Done')

model = dpo_train_loop(model, model_ref, train_loader, valid_loader, optimizer, device=device, num_epochs=args.epochs, stage=2)
print('Stage 2 Done')

for n, p in model.named_parameters():
    checkpoint['model'][n] = p
torch.save(checkpoint, args.output_path)
