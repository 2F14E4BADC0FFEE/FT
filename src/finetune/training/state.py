"""Shared training state. Extracted verbatim from notebook cell 27.

A single shared dict; mutate contents only, never rebind (F4)."""

train_state = {
    "model": None,
    "tokenizer": None,
}
