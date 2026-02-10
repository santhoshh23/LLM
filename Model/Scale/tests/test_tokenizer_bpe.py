import os, tempfile
from tokenizer_bpe import BPETokenizer

def test_bpe_train_save_load_roundtrip():
    from tokenizer_bpe import BPETokenizer
    text = "hello world"
    tok = BPETokenizer()
    tok.train_from_iterator([text], vocab_size=100)
    encoded = tok.encode(text)
    print("ENCODED:", encoded)
    decoded = tok.decode(encoded)
    print("DECODED:", decoded)
    assert decoded == text
