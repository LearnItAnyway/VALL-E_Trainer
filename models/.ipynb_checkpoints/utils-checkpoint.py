from pathlib import Path
from typing import List, Tuple

import re

import numpy as np
import torch
import torch.nn as nn

from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

Symbol = TypeVar('Symbol')


from typing import Any, Dict, List, Optional, Pattern, Union
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.language_switch import LanguageSwitch
from phonemizer.backend.espeak.words_mismatch import WordMismatch
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

from encodec import EncodecModel
from encodec.modules import SConv1d
from encodec.utils import convert_audio
from encodec.modules.seanet import SConvTranspose1d, SEANetResnetBlock

def remove_encodec_weight_norm(model):
    from torch.nn.utils import remove_weight_norm

    encoder = model.encoder.model
    for key in encoder._modules:
        if isinstance(encoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(encoder._modules[key].shortcut.conv.conv)
            block_modules = encoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(encoder._modules[key], SConv1d):
            remove_weight_norm(encoder._modules[key].conv.conv)

    decoder = model.decoder.model
    for key in decoder._modules:
        if isinstance(decoder._modules[key], SEANetResnetBlock):
            remove_weight_norm(decoder._modules[key].shortcut.conv.conv)
            block_modules = decoder._modules[key].block._modules
            for skey in block_modules:
                if isinstance(block_modules[skey], SConv1d):
                    remove_weight_norm(block_modules[skey].conv.conv)
        elif isinstance(decoder._modules[key], SConvTranspose1d):
            remove_weight_norm(decoder._modules[key].convtr.convtr)
        elif isinstance(decoder._modules[key], SConv1d):
            remove_weight_norm(decoder._modules[key].conv.conv)


class AudioTokenizer:
    """EnCodec audio."""

    def __init__(self, device='cuda'):
        # Instantiate a pretrained EnCodec model
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
        remove_encodec_weight_norm(model)

        if not device:
            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda:0")

        self._device = device

        self.codec = model.to(device)
        self.sample_rate = model.sample_rate
        self.channels = model.channels

    @property
    def device(self):
        return self._device

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        return self.codec.encode(wav.to(self.device))

    def decode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.codec.decode(frames)
class TextTokenizer:
    """Phonemize Text."""

    def __init__(
        self,
        language="en-us",
        separator=Separator(word="_", syllable="-", phone="|"),
        preserve_punctuation=True,
        punctuation_marks: Union[str, Pattern] = Punctuation.default_marks(),
        with_stress: bool = False,
        tie: Union[bool, str] = False,
        language_switch: LanguageSwitch = "keep-flags",
        words_mismatch: WordMismatch = "ignore",
    ) -> None:
        phonemizer = EspeakBackend(
            language,
            punctuation_marks=punctuation_marks,
            preserve_punctuation=preserve_punctuation,
            with_stress=with_stress,
            tie=tie,
            language_switch=language_switch,
            words_mismatch=words_mismatch,
        )

        self.backend = phonemizer
        self.separator = separator

    def to_list(self, phonemized: str) -> List[str]:
        fields = []
        for word in phonemized.split(self.separator.word):
            # "ɐ    m|iː|n?"    ɹ|ɪ|z|ɜː|v; h|ɪ|z.
            pp = re.findall(r"\w+|[^\w\s]", word, re.UNICODE)
            fields.extend(
                [p for p in pp if p != self.separator.phone]
                + [self.separator.word]
            )
        assert len("".join(fields[:-1])) == len(phonemized) - phonemized.count(
            self.separator.phone
        )
        return fields[:-1]

    def __call__(self, text, strip=True) -> List[List[str]]:
        text = [text.strip()]
        if isinstance(text, str):
            text = [text]

        phonemized = self.backend.phonemize(
            text, separator=self.separator, strip=strip, njobs=1
        )
        return [self.to_list(p) for p in phonemized][0]

class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")

# Disable __repr__ otherwise it could freeze e.g. Jupyter.
@dataclass(repr=False)
class SymbolTable(Generic[Symbol]):
    '''SymbolTable that maps symbol IDs, found on the FSA arcs to
    actual objects. These objects can be arbitrary Python objects
    that can serve as keys in a dictionary (i.e. they need to be
    hashable and immutable).

    The SymbolTable can only be read to/written from disk if the
    symbols are strings.
    '''
    _id2sym: Dict[int, Symbol] = field(default_factory=dict)
    '''Map an integer to a symbol.
    '''

    _sym2id: Dict[Symbol, int] = field(default_factory=dict)
    '''Map a symbol to an integer.
    '''

    _next_available_id: int = 1
    '''A helper internal field that helps adding new symbols
    to the table efficiently.
    '''

    eps: Symbol = '<eps>'
    '''Null symbol, always mapped to index 0.
    '''

    def __post_init__(self):
        for idx, sym in self._id2sym.items():
            assert self._sym2id[sym] == idx
            assert idx >= 0

        for sym, idx in self._sym2id.items():
            assert idx >= 0
            assert self._id2sym[idx] == sym

        if 0 not in self._id2sym:
            self._id2sym[0] = self.eps
            self._sym2id[self.eps] = 0
        else:
            assert self._id2sym[0] == self.eps
            assert self._sym2id[self.eps] == 0

        self._next_available_id = max(self._id2sym) + 1

    @staticmethod
    def from_str(s: str) -> 'SymbolTable':
        '''Build a symbol table from a string.

        The string consists of lines. Every line has two fields separated
        by space(s), tab(s) or both. The first field is the symbol and the
        second the integer id of the symbol.

        Args:
          s:
            The input string with the format described above.
        Returns:
          An instance of :class:`SymbolTable`.
        '''
        id2sym: Dict[int, str] = dict()
        sym2id: Dict[str, int] = dict()

        for line in s.split('\n'):
            fields = line.split()
            if len(fields) == 0:
                continue  # skip empty lines
            assert len(fields) == 2, \
                    f'Expect a line with 2 fields. Given: {len(fields)}'
            sym, idx = fields[0], int(fields[1])
            assert sym not in sym2id, f'Duplicated symbol {sym}'
            assert idx not in id2sym, f'Duplicated id {idx}'
            id2sym[idx] = sym
            sym2id[sym] = idx

        eps = id2sym.get(0, '<eps>')

        return SymbolTable(_id2sym=id2sym, _sym2id=sym2id, eps=eps)

    @staticmethod
    def from_file(filename: str) -> 'SymbolTable':
        '''Build a symbol table from file.

        Every line in the symbol table file has two fields separated by
        space(s), tab(s) or both. The following is an example file:

        .. code-block::

            <eps> 0
            a 1
            b 2
            c 3

        Args:
          filename:
            Name of the symbol table file. Its format is documented above.

        Returns:
          An instance of :class:`SymbolTable`.

        '''
        with open(filename, 'r', encoding='utf-8') as f:
            return SymbolTable.from_str(f.read().strip())

    def to_str(self) -> str:
        '''
        Returns:
          Return a string representation of this object. You can pass
          it to the method ``from_str`` to recreate an identical object.
        '''
        s = ''
        for idx, symbol in sorted(self._id2sym.items()):
            s += f'{symbol} {idx}\n'
        return s

    def to_file(self, filename: str):
        '''Serialize the SymbolTable to a file.

        Every line in the symbol table file has two fields separated by
        space(s), tab(s) or both. The following is an example file:

        .. code-block::

            <eps> 0
            a 1
            b 2
            c 3

        Args:
          filename:
            Name of the symbol table file. Its format is documented above.
        '''
        with open(filename, 'w') as f:
            for idx, symbol in sorted(self._id2sym.items()):
                print(symbol, idx, file=f)

    def add(self, symbol: Symbol, index: Optional[int] = None) -> int:
        '''Add a new symbol to the SymbolTable.

        Args:
            symbol:
                The symbol to be added.
            index:
                Optional int id to which the symbol should be assigned.
                If it is not available, a ValueError will be raised.

        Returns:
            The int id to which the symbol has been assigned.
        '''
        # Already in the table? Return its ID.
        if symbol in self._sym2id:
            return self._sym2id[symbol]
        # Specific ID not provided - use next available.
        if index is None:
            index = self._next_available_id
        # Specific ID provided but not available.
        if index in self._id2sym:
            raise ValueError(f"Cannot assign id '{index}' to '{symbol}' - "
                             f"already occupied by {self._id2sym[index]}")
        self._sym2id[symbol] = index
        self._id2sym[index] = symbol

        # Update next available ID if needed
        if self._next_available_id <= index:
            self._next_available_id = index + 1

        return index

    def get(self, k: Union[int, Symbol]) -> Union[Symbol, int]:
        '''Get a symbol for an id or get an id for a symbol

        Args:
          k:
            If it is an id, it tries to find the symbol corresponding
            to the id; if it is a symbol, it tries to find the id
            corresponding to the symbol.

        Returns:
          An id or a symbol depending on the given `k`.
        '''
        if isinstance(k, int):
            return self._id2sym[k]
        else:
            return self._sym2id[k]

    def merge(self, other: 'SymbolTable') -> 'SymbolTable':
        '''Create a union of two SymbolTables.
        Raises an AssertionError if the same IDs are occupied by
        different symbols.

        Args:
            other:
                A symbol table to merge with ``self``.

        Returns:
            A new symbol table.
        '''
        self._check_compatible(other)

        id2sym = {**self._id2sym, **other._id2sym}
        sym2id = {**self._sym2id, **other._sym2id}

        return SymbolTable(_id2sym=id2sym, _sym2id=sym2id, eps=self.eps)

    def _check_compatible(self, other: 'SymbolTable') -> None:
        # Epsilon compatibility
        assert self.eps == other.eps, f'Mismatched epsilon symbol: ' \
                                      f'{self.eps} != {other.eps}'
        # IDs compatibility
        common_ids = set(self._id2sym).intersection(other._id2sym)
        for idx in common_ids:
            assert self[idx] == other[idx], f'ID conflict for id: {idx}, ' \
                                            f'self[idx] = "{self[idx]}", ' \
                                            f'other[idx] = "{other[idx]}"'
        # Symbols compatibility
        common_symbols = set(self._sym2id).intersection(other._sym2id)
        for sym in common_symbols:
            assert self[sym] == other[sym], f'ID conflict for id: {sym}, ' \
                                            f'self[sym] = "{self[sym]}", ' \
                                            f'other[sym] = "{other[sym]}"'

    def __getitem__(self, item: Union[int, Symbol]) -> Union[Symbol, int]:
        return self.get(item)

    def __contains__(self, item: Union[int, Symbol]) -> bool:
        if isinstance(item, int):
            return item in self._id2sym
        else:
            return item in self._sym2id

    def __len__(self) -> int:
        return len(self._id2sym)

    def __eq__(self, other: 'SymbolTable') -> bool:
        if len(self) != len(other):
            return False

        for s in self.symbols:
            if self[s] != other[s]:
                return False

        return True

    @property
    def ids(self) -> List[int]:
        '''Returns a list of integer IDs corresponding to the symbols.
        '''
        ans = list(self._id2sym.keys())
        ans.sort()
        return ans

    @property
    def symbols(self) -> List[Symbol]:
        '''Returns a list of symbols (e.g., strings) corresponding to
        the integer IDs.
        '''
        ans = list(self._sym2id.keys())
        ans.sort()
        return ans

class TextTokenCollater:
    """Collate list of text tokens

    Map sentences to integers. Sentences are padded to equal length.
    Beginning and end-of-sequence symbols can be added.

    Example:
        >>> token_collater = TextTokenCollater(text_tokens)
        >>> tokens_batch, tokens_lens = token_collater(text)

    Returns:
        tokens_batch: IntTensor of shape (B, L)
            B: batch dimension, number of input sentences
            L: length of the longest sentence
        tokens_lens: IntTensor of shape (B,)
            Length of each sentence after adding <eos> and <bos>
            but before padding.
    """

    def __init__(
        self,
        text_tokens: List[str],
        add_eos: bool = True,
        add_bos: bool = True,
        pad_symbol: str = "<pad>",
        bos_symbol: str = "<bos>",
        eos_symbol: str = "<eos>",
    ):
        self.pad_symbol = pad_symbol

        self.add_eos = add_eos
        self.add_bos = add_bos

        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol

        unique_tokens = (
            [pad_symbol]
            + ([bos_symbol] if add_bos else [])
            + ([eos_symbol] if add_eos else [])
            + sorted(text_tokens)
        )

        self.token2idx = {token: idx for idx, token in enumerate(unique_tokens)}
        self.idx2token = [token for token in unique_tokens]

    def __call__(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_seqs = [[p for p in text] for text in texts]
        max_len = len(max(tokens_seqs, key=len))

        seqs = [
            ([self.bos_symbol] if self.add_bos else [])
            + list(seq)
            + ([self.eos_symbol] if self.add_eos else [])
            + [self.pad_symbol] * (max_len - len(seq))
            for seq in tokens_seqs
        ]

        tokens_batch = torch.from_numpy(
            np.array(
                [[self.token2idx[token] for token in seq] for seq in seqs],
                dtype=np.int64,
            )
        )

        tokens_lens = torch.IntTensor(
            [
                len(seq) + int(self.add_eos) + int(self.add_bos)
                for seq in tokens_seqs
            ]
        )

        return tokens_batch, tokens_lens

def get_text_token_collater(text_tokens_file: str) -> TextTokenCollater:
    text_tokens_path = Path(text_tokens_file)
    unique_tokens = SymbolTable.from_file(text_tokens_path)
    collater = TextTokenCollater(
        unique_tokens.symbols, add_bos=True, add_eos=True
    )
    return collater

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)

class Transpose(nn.Identity):
    """(N, T, D) -> (N, D, T)"""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.transpose(1, 2)



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
        else:
            self.best_loss = val_loss
            self.counter = 0
        if self.counter >= self.patience:
            self.early_stop = True

