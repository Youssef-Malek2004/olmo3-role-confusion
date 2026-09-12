"""Model loading, chat rendering, batched greedy generation, and prompt-end activation capture.

Conventions (see docs/archive/PROJECT_PLAN.md section 10):
  * The prompt is rendered with the checkpoint's own chat template and add_generation_prompt=True.
    Whatever fixed assistant/thinking opener the template appends is part of the *input*.
  * Activations are taken from a separate forward pass over the padded prompt batch with
    output_hidden_states=True. ``hidden_states[0]`` is the embedding output; ``hidden_states[k]``
    is the output of transformer block k counting from one. Block numbers in configs are one-based.
  * Left padding is used so the final non-padding input token is at position -1 for every row.
    This is asserted, not assumed.
  * Only the continuation (tokens after the prompt) is decoded and passed to the parser.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LoadedModel:
    repo_id: str
    revision: str
    model: torch.nn.Module
    tokenizer: object
    device: str
    dtype: str
    num_layers: int
    hidden_size: int
    chat_template_sha1: str
    eos_token_ids: tuple[int, ...]


def pick_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(repo_id: str, revision: str, device: str | None = None,
               dtype: torch.dtype = torch.bfloat16) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    tok = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo_id, revision=revision, dtype=dtype)
    model.to(device)
    model.eval()
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tok.eos_token_id
    eos_ids = tuple(eos) if isinstance(eos, (list, tuple)) else (int(eos),)
    template = tok.chat_template or ""
    return LoadedModel(
        repo_id=repo_id, revision=revision, model=model, tokenizer=tok, device=device,
        dtype=str(dtype).replace("torch.", ""),
        num_layers=int(model.config.num_hidden_layers), hidden_size=int(model.config.hidden_size),
        chat_template_sha1=hashlib.sha1(template.encode("utf-8")).hexdigest(),
        eos_token_ids=eos_ids,
    )


def render_chat(lm: LoadedModel, messages: list[dict]) -> str:
    """Apply the checkpoint's chat template to a prepared message list with the generation prompt."""
    return lm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@dataclass
class GenerationResult:
    rendered_prompt: str
    prompt_token_count: int
    generated_text: str
    generated_token_count: int
    finished: bool
    prompt_end_activations: np.ndarray  # shape (n_blocks, hidden_size), float32
    seconds: float
    span_mean_activations: dict[str, np.ndarray] | None = None  # name -> (n_blocks, hidden), mean over span tokens


@dataclass
class Steer:
    """Add ``alpha * vector`` to the residual stream after block ``block_one_based`` at masked prompt positions.

    Applied only during the prefill forward (sequence length > 1); decode steps are untouched. This is the
    "constant vector depending on which turn the token is in" intervention. ``masks`` is (B, T) bool over the
    padded prompt positions.
    """
    block_one_based: int
    vector: np.ndarray
    alpha: float
    masks: np.ndarray


def span_token_mask(tokenizer, rendered_prompt: str, start_char: int, end_char: int, padded_len: int) -> np.ndarray:
    """Boolean mask over left-padded token positions for tokens overlapping [start_char, end_char)."""
    enc = tokenizer(rendered_prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    n = len(offsets)
    mask = np.zeros(padded_len, dtype=bool)
    pad = padded_len - n
    for i, (a, b) in enumerate(offsets):
        if b > start_char and a < end_char and b > a:
            mask[pad + i] = True
    return mask


def find_span(rendered_prompt: str, needle: str) -> tuple[int, int]:
    i = rendered_prompt.find(needle)
    if i < 0:
        raise ValueError("span text not found in rendered prompt")
    return i, i + len(needle)


@dataclass(frozen=True)
class Decoding:
    """Decoding policy. ``greedy`` ignores temperature/top_p. ``seed`` is applied per batch."""
    mode: str = "greedy"  # "greedy" | "sample"
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None

    def to_dict(self) -> dict:
        return {"mode": self.mode, "temperature": self.temperature, "top_p": self.top_p, "seed": self.seed}


def batch_seed(base_seed: int, variant: str, draw: int, question_ids: list[str]) -> int:
    """Deterministic 31-bit seed from the run seed, variant, draw index, and the batch's question IDs.

    The same batch composition therefore gets the same seed at every checkpoint. This buys
    reproducibility on one machine/library stack, not paired comparability: different models
    consume the random stream differently.
    """
    key = f"{base_seed}|{variant}|{draw}|" + ",".join(question_ids)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


@torch.no_grad()
def generate_batch(lm: LoadedModel, rendered_prompts: list[str], max_new_tokens: int,
                   blocks_one_based: list[int], static_cache: bool = True,
                   decoding: Decoding | None = None, steer: Steer | None = None,
                   span_masks: dict[str, np.ndarray] | None = None) -> list[GenerationResult]:
    """``span_masks`` maps a name to a (B, T) bool mask; mean activations over each span are returned per row."""
    decoding = decoding or Decoding()
    tok = lm.tokenizer
    hook_handle = None
    if steer is not None:
        layer = lm.model.model.layers[steer.block_one_based - 1]
        vec = torch.tensor(steer.vector, dtype=torch.bfloat16 if lm.dtype == "bfloat16" else torch.float32,
                           device=lm.device) * float(steer.alpha)
        mask_t = torch.tensor(steer.masks, device=lm.device)

        def _hook(module, args, output):
            hs = output[0] if isinstance(output, tuple) else output
            if hs.shape[1] != mask_t.shape[1]:
                return None  # decode step (or a shape we did not plan for): leave untouched
            hs = hs + mask_t.unsqueeze(-1).to(hs.dtype) * vec
            return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs

        hook_handle = layer.register_forward_hook(_hook)
    # The template already contains the special tokens, so do not add BOS/EOS again.
    enc = tok(rendered_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc["input_ids"].to(lm.device)
    attention_mask = enc["attention_mask"].to(lm.device)
    if not bool(attention_mask[:, -1].all()):
        raise RuntimeError("left padding expected: last input position must be non-padding")
    prompt_lens = attention_mask.sum(dim=1).tolist()

    t0 = time.time()
    # 1) Prompt-end activations from a prompt-only forward pass.
    out = lm.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                   use_cache=False)
    hs = out.hidden_states
    if len(hs) != lm.num_layers + 1:
        raise RuntimeError(f"expected {lm.num_layers + 1} hidden states, got {len(hs)}")
    for b in blocks_one_based:
        if not 1 <= b <= lm.num_layers:
            raise ValueError(f"block {b} out of range 1..{lm.num_layers}")
    acts = torch.stack([hs[b][:, -1, :] for b in blocks_one_based], dim=1)  # (B, n_blocks, H)
    acts = acts.float().cpu().numpy()
    span_means: dict[str, np.ndarray] = {}
    if span_masks:
        for name, m in span_masks.items():
            mt = torch.tensor(m, device=lm.device).unsqueeze(-1).to(hs[0].dtype)  # (B, T, 1)
            denom = mt.sum(dim=1).clamp(min=1)  # (B, 1)
            span_means[name] = torch.stack([(hs[b] * mt).sum(dim=1) / denom for b in blocks_one_based],
                                           dim=1).float().cpu().numpy()
    del out, hs

    # 2) Greedy generation. A static cache is preallocated once for the full length instead of
    #    growing by concatenation every step, which fragments the MPS allocator and balloons memory.
    gen_kwargs = dict(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
        eos_token_id=list(lm.eos_token_ids), num_beams=1, top_k=None,
    )
    if decoding.mode == "sample":
        if decoding.seed is None:
            raise ValueError("sampling requires a seed")
        torch.manual_seed(decoding.seed)
        gen_kwargs.update(do_sample=True, temperature=decoding.temperature, top_p=decoding.top_p)
    elif decoding.mode == "greedy":
        gen_kwargs.update(do_sample=False, temperature=None, top_p=None)
    else:
        raise ValueError(f"unknown decoding mode {decoding.mode}")
    if static_cache:
        gen_kwargs["cache_implementation"] = "static"
    import os
    if os.environ.get("GEN_HEARTBEAT"):
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _Heartbeat(StoppingCriteria):
            def __init__(self, every: int):
                self.every, self.i, self.t = every, 0, time.time()

            def __call__(self, ids, scores, **kw):
                self.i += 1
                if self.i % self.every == 0:
                    print(f"    [heartbeat] step {self.i} len {ids.shape[1]} {time.time() - self.t:.0f}s", flush=True)
                return torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)

        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_Heartbeat(int(os.environ["GEN_HEARTBEAT"]))])
    try:
        gen = lm.model.generate(**gen_kwargs)
    finally:
        if hook_handle is not None:
            hook_handle.remove()
    seconds = time.time() - t0
    cont = gen[:, input_ids.shape[1]:].cpu()
    del gen
    empty_cache(lm.device)

    results = []
    eos_set = set(lm.eos_token_ids)
    for i in range(cont.shape[0]):
        ids = cont[i].tolist()
        n_gen = len(ids)
        finished = False
        for j, t in enumerate(ids):
            if t in eos_set:
                n_gen = j + 1
                finished = True
                break
        # Everything after the first EOS is padding; strip it. Exclude the EOS token from text.
        text_ids = ids[: n_gen - 1] if finished else ids[:n_gen]
        text = tok.decode(text_ids, skip_special_tokens=False)
        results.append(GenerationResult(
            rendered_prompt=rendered_prompts[i], prompt_token_count=int(prompt_lens[i]),
            generated_text=text, generated_token_count=int(n_gen), finished=finished,
            prompt_end_activations=acts[i], seconds=seconds / cont.shape[0],
            span_mean_activations={k: v[i] for k, v in span_means.items()} if span_means else None,
        ))
    return results


def empty_cache(device: str) -> None:
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def device_memory_gb(device: str) -> dict:
    """Allocated/driver memory in GB for logging. Zero on CPU."""
    if device == "mps":
        return {"allocated_gb": torch.mps.current_allocated_memory() / 2**30,
                "driver_gb": torch.mps.driver_allocated_memory() / 2**30}
    if device == "cuda":
        return {"allocated_gb": torch.cuda.memory_allocated() / 2**30,
                "driver_gb": torch.cuda.memory_reserved() / 2**30}
    return {"allocated_gb": 0.0, "driver_gb": 0.0}


def sync_device(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def prefill_hidden_states(lm: LoadedModel, rendered_prompts: list[str], blocks_one_based: list[int]):
    """Prompt-only forward pass. Returns (hidden (B, T, n_blocks, H) float16 numpy, attention_mask (B, T) numpy).

    Left padding, so position T-1 is the last real token for every row.
    """
    tok = lm.tokenizer
    enc = tok(rendered_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc["input_ids"].to(lm.device)
    attention_mask = enc["attention_mask"].to(lm.device)
    out = lm.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    stacked = torch.stack([hs[b] for b in blocks_one_based], dim=2)  # (B, T, n_blocks, H)
    result = stacked.to(torch.float16).cpu().numpy()
    del out, hs, stacked
    empty_cache(lm.device)
    return result, enc["attention_mask"].numpy()
