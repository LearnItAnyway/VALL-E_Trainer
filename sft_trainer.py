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
parser.add_argument('--token_path', type=str, default='unique_text_tokens_v1.k2symbols')
parser.add_argument('--output_path', type=str, default='tmp.pt')

parser.add_argument('--data_dir', type=str, default='data_sft', help='Directory containing audio files')
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
model = get_model(AttributeDict(checkpoint))
missing_keys, unexpected_keys = model.load_state_dict(
    checkpoint["model"], strict=True
)
assert not missing_keys

text_collater = get_text_token_collater(args.token_path)
audio_tokenizer = AudioTokenizer(device)

model.to(device)

class WavTextDataset(Dataset):
    def __init__(self, data_dir):
        self.audio_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.wav')]
        self.text_paths = [os.path.join(data_dir, os.path.splitext(os.path.basename(fname))[0] + '.txt') for fname in self.audio_paths]

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        text_path = self.text_paths[idx]

        with open(text_path, 'r') as f:
            text_prompt = f.read().strip()

        wav_pr, sr = torchaudio.load(audio_path)
        wav_pr = convert_audio(wav_pr, sr, audio_tokenizer.sample_rate, audio_tokenizer.channels)
        return text_prompt, wav_pr

def get_input(wav_pr, text_prompt):
    audio_prompts = audio_tokenizer.encode(wav_pr.unsqueeze(0))[0][0].transpose(2, 1).to(device)
    text_tokens, text_tokens_lens = text_collater([text_tokenizer(text_prompt)])
    return text_tokens, text_tokens_lens, audio_prompts

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

# Early Stopping Class

def simple_train_loop(model, train_loader, valid_loader, optimizer, device, num_epochs=10, stage=0):
    # Send model to device
    model = model.to(device)
    early_stopping = EarlyStopping(patience=3)
    # Loop over epochs
    if stage==2: nars = np.arange(1, 8)
    else: nars = [0]
    for epoch in range(num_epochs):
        loss_sum = 0
        optimizer.zero_grad()
        model.eval()
        for _, batch in enumerate(train_loader):
            for nar_ in nars :
                text_prompt, wav_pr = batch
                text_tokens, _, audio_features = get_input(wav_pr[0], text_prompt[0])
                text_tokens_lens = torch.LongTensor([text_tokens.shape[1]]).to(device)
                audio_features_lens = torch.LongTensor([audio_features.shape[1]]).to(device)
                # Do something with the data
                assert text_tokens.ndim == 2 and  audio_features.ndim == 3

                with torch.set_grad_enabled(True):
                    predicts, loss, metrics = model(
                        x=text_tokens.to(device),
                        x_lens=text_tokens_lens,
                        y=audio_features,
                        y_lens=audio_features_lens,
                        train_stage=stage,
                        nar_=nar_
                    )

                loss.backward()
                loss_sum += loss.item()
        optimizer.step()

        loss_sum_ = 0
        model.eval()
        for _, batch in enumerate(valid_loader):
            for nar_ in nars :
                text_prompt, wav_pr = batch
                text_tokens, _, audio_features = get_input(wav_pr[0], text_prompt[0])
                text_tokens_lens = torch.LongTensor([text_tokens.shape[1]]).to(device)
                audio_features_lens = torch.LongTensor([audio_features.shape[1]]).to(device)

                # Forward pass
                with torch.set_grad_enabled(False):
                    predicts, loss, metrics = model(
                        x=text_tokens.to(device),
                        x_lens=text_tokens_lens,
                        y=audio_features,
                        y_lens=audio_features_lens,
                        train_stage=stage,
                        nar_=nar_
                    )

                loss_sum_ += loss.item()

        loss_sum /= len(train_loader) * (len(nars))
        loss_sum_ /= len(valid_loader) * (len(nars))
        print("Train Loss : ", loss_sum, "Valid Loss : ", loss_sum_)
        early_stopping(loss_sum_)
        if early_stopping.early_stop:
            break

    return model
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Call the simple training loop
model = simple_train_loop(model, train_loader, valid_loader, optimizer, device=device, num_epochs=args.epochs, stage=2)
print('Stage 2 Done')
model = simple_train_loop(model, train_loader, valid_loader, optimizer, device=device, num_epochs=args.epochs, stage=1)
print('Stage 1 Done')

for n, p in model.named_parameters():
    checkpoint['model'][n] = p
torch.save(checkpoint, args.output_path)
