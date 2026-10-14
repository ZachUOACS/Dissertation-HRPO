"""Weight-only quantisation study (weight_quant.py).

What must hold for the study to mean anything:
  * RTN keeps at most 2^b levels per group, errs by at most half a step, and is idempotent;
  * only the 7 block projections per layer change; embeddings, norms and anything else do not;
  * a projection count other than 7 per layer is an error, never a silent bf16 evaluation;
  * merging LoRA leaves the function unchanged (the bf16 scheme is merge only), and the
    quantised scheme then acts on the merged weight;
  * NF4 matches bitsandbytes' own 4-bit layer (GPU only).

    python -m pytest tests/test_weight_quant.py -q
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import weight_quant as wq  # noqa: E402


class Block(nn.Module):
    def __init__(self, d, f):
        super().__init__()
        self.self_attn = nn.Module()
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, n, nn.Linear(d, d, bias=n != "o_proj"))
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(d, f, bias=False)
        self.mlp.up_proj = nn.Linear(d, f, bias=False)
        self.mlp.down_proj = nn.Linear(f, d, bias=False)
        self.input_layernorm = nn.LayerNorm(d)


class Config:
    """The two attributes used here; peft reads the config through to_dict()."""
    def __init__(self, n):
        self.num_hidden_layers = n
        self.tie_word_embeddings = False

    def to_dict(self):
        return dict(vars(self))


class Tiny(nn.Module):
    def __init__(self, d=128, f=256, n=2):
        super().__init__()
        self.config = Config(n)
        self.embed_tokens = nn.Embedding(50, d)
        self.layers = nn.ModuleList(Block(d, f) for _ in range(n))
        self.thinking_residual_gate_r = nn.Linear(d, d)       # an HRPO gate: must stay untouched

    def forward(self, x):
        h = self.embed_tokens(x)
        for b in self.layers:
            a = b.self_attn
            h = h + a.o_proj(a.q_proj(h) * torch.sigmoid(a.k_proj(h)) + a.v_proj(h))
            m = b.mlp
            h = h + m.down_proj(torch.nn.functional.silu(m.gate_proj(h)) * m.up_proj(h))
        return h


def test_effective_bits():
    assert wq.effective_bits("bf16") == 16
    assert wq.effective_bits("int4") == pytest.approx(4 + 20 / 128)
    assert wq.effective_bits("int8") == pytest.approx(8 + 24 / 128)
    assert wq.effective_bits("nf4") == pytest.approx(4.127, abs=1e-3)   # QLoRA's figure
    with pytest.raises(ValueError):
        wq.effective_bits("int2")


@pytest.mark.parametrize("bits", [3, 4, 8])
def test_rtn_levels_error_and_idempotence(bits):
    torch.manual_seed(0)
    w = torch.randn(16, 512) * 0.02
    q = wq.rtn_quantise(w, bits)
    assert q.shape == w.shape and q.dtype == w.dtype
    groups = q.reshape(16, 4, 128)
    for row in groups.reshape(-1, 128):
        assert row.unique().numel() <= 2 ** bits
    g = w.reshape(16, 4, 128)
    step = (g.amax(-1).clamp(min=0) - g.amin(-1).clamp(max=0)) / (2 ** bits - 1)
    err = (q - w).abs().reshape(16, 4, 128).amax(-1)
    assert (err <= step * 0.5 * 1.01 + 1e-7).all()
    assert torch.allclose(wq.rtn_quantise(q, bits), q, atol=1e-7)
    if bits == 8:
        assert ((q - w).norm() / w.norm()) < 0.01


def test_rtn_constant_group_and_shape_errors():
    w = torch.zeros(2, 128)
    assert torch.equal(wq.rtn_quantise(w, 4), w)
    for bits in (3, 4, 8):
        c = torch.full((2, 128), 0.3)                         # one-signed group: range [0, 0.3]
        assert torch.allclose(wq.rtn_quantise(c, bits), c, atol=1e-7)
        assert torch.allclose(wq.rtn_quantise(-c, bits), -c, atol=1e-7)
    small = torch.randn(2, 128) * 1e-6                        # tiny range: no fp16 underflow
    q = wq.rtn_quantise(small, 8)
    assert torch.isfinite(q).all() and ((q - small).norm() / small.norm()) < 0.01
    with pytest.raises(ValueError):
        wq.rtn_quantise(torch.randn(2, 100), 4)


def test_only_block_projections_change():
    torch.manual_seed(0)
    model = Tiny()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    stats = wq.apply_weight_quant(model, "int4")
    assert stats["projections"] == 14 and stats["lora_merged"] == 0
    for n, p in model.named_parameters():
        is_proj = wq.TARGET_RE.search(n.rsplit(".", 1)[0]) is not None and n.endswith("weight")
        assert torch.equal(p, before[n]) != is_proj, n
    assert stats["weight_gb"] < stats["weight_gb_bf16"]


def test_bf16_scheme_is_a_no_op_without_lora():
    model = Tiny()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    wq.apply_weight_quant(model, "bf16")
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_wrong_projection_count_is_an_error():
    model = Tiny()
    with pytest.raises(RuntimeError):
        wq.apply_weight_quant(model, "int4", num_layers=3)


def test_target_pattern():
    assert wq.TARGET_RE.search("base_model.model.model.layers.11.mlp.down_proj")
    assert wq.TARGET_RE.search("model.layers.0.self_attn.q_proj")
    assert not wq.TARGET_RE.search("model.layers.0.self_attn.q_proj.base_layer")
    assert not wq.TARGET_RE.search("model.layers.0.self_attn.q_proj.lora_A.default")
    assert not wq.TARGET_RE.search("model.thinking_residual_gate_r")
    assert not wq.TARGET_RE.search("lm_head")


def test_lora_merge_preserves_function_then_quantises_merged_weight():
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    model = Tiny()
    cfg = peft.LoraConfig(r=8, lora_alpha=16, target_modules=list(wq.PROJECTIONS))
    model = peft.inject_adapter_in_model(cfg, model)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            nn.init.normal_(p, std=0.05)                      # a trained, non-zero adapter
    model.eval()
    x = torch.randint(0, 50, (3, 7))
    with torch.no_grad():
        y_lora = model(x)
    stats = wq.apply_weight_quant(model, "bf16")
    assert stats["lora_merged"] == 14
    with torch.no_grad():
        y_merged = model(x)
    assert torch.allclose(y_lora, y_merged, rtol=1e-4, atol=1e-5)
    # quantising now acts on W + BA, and the LoRA path stays off (merged)
    q_proj = model.layers[0].self_attn.q_proj
    merged_w = q_proj.base_layer.weight.detach().clone()
    wq.apply_weight_quant(model, "int4")
    assert torch.allclose(q_proj.base_layer.weight, wq.rtn_quantise(merged_w, 4))
    assert q_proj.merged


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bitsandbytes NF4 needs CUDA")
def test_nf4_matches_bitsandbytes_layer():
    bnb = pytest.importorskip("bitsandbytes")
    torch.manual_seed(0)
    w = (torch.randn(256, 512) * 0.02).to(torch.bfloat16).cuda()
    ours = wq.nf4_quantise(w)
    layer = bnb.nn.Linear4bit(512, 256, bias=False, compute_dtype=torch.bfloat16,
                              compress_statistics=True, quant_type="nf4")
    layer.weight = bnb.nn.Params4bit(w.cpu(), requires_grad=False, compress_statistics=True,
                                     quant_type="nf4", blocksize=64)
    layer = layer.cuda()
    ref = bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state).to(torch.bfloat16)
    assert torch.equal(ours, ref)
    assert ((ours.float() - w.float()).norm() / w.float().norm()) < 0.15
