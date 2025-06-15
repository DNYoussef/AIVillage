from pathlib import Path
from tiktoken.load import load_tiktoken_bpe
from tiktoken import Encoding

TOKEN_FILE = Path(__file__).resolve().parent / "token_data" / "cl100k_base.tiktoken"


def get_cl100k_encoding() -> Encoding:
    """Load the cl100k_base tokenizer using the bundled BPE file."""
    mergeable_ranks = load_tiktoken_bpe(str(TOKEN_FILE))
    special_tokens = {
        "<|endoftext|>": 100257,
        "<|fim_prefix|>": 100258,
        "<|fim_middle|>": 100259,
        "<|fim_suffix|>": 100260,
        "<|endofprompt|>": 100276,
    }
    return Encoding(
        name="cl100k_base",
        pat_str=r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s",
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
