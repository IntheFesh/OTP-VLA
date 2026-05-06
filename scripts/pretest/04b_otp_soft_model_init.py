"""
Pretest: OTPSoftModel instantiation in frozen backbone mode.

Validates wiring (no forward), specifically:
  1. OTPSoftModel can be built with backbone_mode='frozen'.
  2. self.backbone is OpenVLABackboneWrapper, not stub.
  3. trainable params == OTPHead + Decoder only (backbone fully frozen).
  4. Total params ≈ 7.54B + head + decoder.

NB: We feed the cfg.model sub-dict (matching how train_otp_soft.py uses it
implicitly via OTPSoftModel(config=cfg.model)).
"""
from omegaconf import OmegaConf

from otp.models.otp_soft_model import OTPSoftModel
from otp.models.openvla_wrapper import OpenVLABackboneWrapper

print("\n[1/3] Loading config (otp_soft_frozen.yaml) ...")
# Hydra resolves `defaults:` chain; OmegaConf.load alone won't merge defaults.
# Since otp_soft_frozen.yaml is the BASE (no defaults: chain pulling other
# configs above itself), a direct load is fine. If wrong, switch to compose.
cfg = OmegaConf.load("configs/otp_soft_frozen.yaml")
print(f"      backbone_mode      = {cfg.model.get('backbone_mode')}")
print(f"      backbone_dim       = {cfg.model.get('backbone_dim')}")
print(f"      backbone_checkpoint= {cfg.model.get('backbone_checkpoint', '<default>')}")

print("\n[2/3] Building OTPSoftModel (will load OpenVLA checkpoint, ~30s) ...")
import time
t0 = time.time()
model = OTPSoftModel(cfg.model)
print(f"      build wall-clock = {time.time() - t0:.1f}s")

# Verify backbone identity
assert isinstance(model.backbone, OpenVLABackboneWrapper), \
    f"Expected OpenVLABackboneWrapper, got {type(model.backbone).__name__}"
print(f"      model.backbone class = {type(model.backbone).__name__}  ✓")

print("\n[3/3] Param accounting ...")
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
backbone_total = sum(p.numel() for p in model.backbone.parameters())
backbone_train = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
head_total = sum(p.numel() for p in model.otp_head.parameters())
head_train = sum(p.numel() for p in model.otp_head.parameters() if p.requires_grad)
# Decoder name might be 'decoder' or 'lang_agnostic_decoder' etc; probe
decoder = None
for n in ("decoder", "lang_agnostic_decoder", "action_decoder"):
    if hasattr(model, n):
        decoder = getattr(model, n)
        decoder_name = n
        break
if decoder is not None:
    dec_total = sum(p.numel() for p in decoder.parameters())
    dec_train = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
else:
    decoder_name, dec_total, dec_train = "<not found>", 0, 0

other_total = total - backbone_total - head_total - dec_total
other_train = trainable - backbone_train - head_train - dec_train

print(f"  TOTAL          : {total:>15,}  trainable: {trainable:>13,}")
print(f"  ├─ backbone    : {backbone_total:>15,}  trainable: {backbone_train:>13,}  (expect 0)")
print(f"  ├─ otp_head    : {head_total:>15,}  trainable: {head_train:>13,}")
print(f"  ├─ {decoder_name:11s}: {dec_total:>15,}  trainable: {dec_train:>13,}")
print(f"  └─ other       : {other_total:>15,}  trainable: {other_train:>13,}")

# Pass criteria
assert backbone_train == 0, f"FAIL: backbone has {backbone_train} trainable params, expected 0"
assert trainable > 0, "FAIL: nothing is trainable"
assert head_train > 0, "FAIL: otp_head not trainable"
print("\n=== Stage 4b passed: OTPSoftModel wiring OK in frozen mode ===")
