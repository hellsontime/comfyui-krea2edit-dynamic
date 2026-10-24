"""Per-region reference weights for the Krea 2 Identity Edit patch.

The upstream patch (comfyui-krea2edit) multiplies target->reference attention by one
factor inside a mask thresholded at 0.5. These nodes keep its whole source path and
replace only that bias with a per-pixel multiplier map, so the face, the hair, the
clothes and the background of the reference can each pull with their own strength.
"""

import math
import sys

import torch
import torch.nn.functional as F

import nodes as comfy_nodes

_TAG = "_krea2_dynamic_weights"
_ALL = "_krea2_dynamic_all_refs"
LEFT = "beside (left of target)"


def _upstream():
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("Krea2EditModelPatch")
    if cls is None:
        raise RuntimeError("comfyui-krea2edit (>= 2.0.0) is not installed; this pack builds on its patch node")
    return cls, sys.modules[cls.__module__]


def _install(module):
    # One dispatcher on the upstream module: the upstream node keeps its own bias, ours
    # is taken only for a mask that carries our tag.
    if getattr(module, "_krea2_dynamic_installed", False):
        return
    upstream_bias = module._ref_attn_bias

    def ref_attn_bias(boosts, boost_mask, txtlen, slens, tgtlen, mask_hw, device, dtype):
        if boost_mask is None or not getattr(boost_mask, _TAG, False):
            return upstream_bias(boosts, boost_mask, txtlen, slens, tgtlen, mask_hw, device, dtype)
        offs = [txtlen]
        for sl in slens:
            offs.append(offs[-1] + sl)
        rows0 = offs[-1]
        L = rows0 + tgtlen
        bias = torch.zeros(1, 1, L, L, device=device, dtype=dtype)
        weighted = range(len(slens)) if getattr(boost_mask, _ALL, False) else [len(slens) - 1]
        for i, b in enumerate(boosts):
            if i not in weighted and b != 1.0:
                bias[:, :, rows0:, offs[i]:offs[i + 1]] = math.log(max(b, 1e-4))
        w = boost_mask[:1]
        if w.ndim == 2:
            w = w[None]
        for i in weighted:
            wi = F.interpolate(w[None].float(), size=mask_hw[i], mode="area")[0, 0]
            bias[:, :, rows0:, offs[i]:offs[i + 1]] = wi.reshape(-1).clamp(min=1e-4).log().to(device=device, dtype=dtype)
        return bias

    upstream_place = module._place_refs

    def place_refs(ref_ids, modes, h_, w_):
        # a reference to the left of the canvas, so one on each side pulls the subject to neither edge
        if isinstance(modes, str):
            modes = [modes] * len(ref_ids)
        left = [m == LEFT for m in modes]
        out = upstream_place(ref_ids, [module.REF_POSITIONS[0] if l else m for l, m in zip(left, modes)], h_, w_)
        x1 = 0.0
        for i in reversed(range(len(out))):
            if left[i]:
                ids = out[i].clone()
                ids[..., 2] = ids[..., 2] - ids[..., 2].max() - 1.0 + x1
                x1 = float(ids[..., 2].min())
                out[i] = ids
        return out

    module._ref_attn_bias = ref_attn_bias
    module._place_refs = place_refs
    module._krea2_dynamic_installed = True


class Krea2RefWeights:
    """Paints a per-pixel multiplier map over the reference: a base weight, then each
    connected mask laid over it with its own weight, later ones on top."""

    SLOTS = 6

    @classmethod
    def INPUT_TYPES(cls):
        opt = {}
        for i in range(1, cls.SLOTS + 1):
            opt[f"mask_{i}"] = ("MASK",)
            opt[f"weight_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05})
        return {"required": {
            "reference": ("IMAGE", {"tooltip": "the reference image the masks belong to; sets the map's size"}),
            "base": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05,
                               "tooltip": "multiplier wherever no mask covers the reference; 1 = as trained"}),
        }, "optional": opt}

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("ref_weights", "preview")
    FUNCTION = "paint"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Per-pixel attention multipliers for the reference: 1 = as trained, 2 = pulls twice as hard, 0.05 = nearly ignored."

    def paint(self, reference, base, **kw):
        h, w = reference.shape[1], reference.shape[2]
        out = torch.full((1, h, w), float(base))
        for i in range(1, self.SLOTS + 1):
            m = kw.get(f"mask_{i}")
            if m is None:
                continue
            m = m[:1].float()
            if m.shape[-2:] != (h, w):
                m = F.interpolate(m[None], size=(h, w), mode="bilinear")[0]
            m = m.clamp(0, 1)
            out = out * (1 - m) + float(kw.get(f"weight_{i}", 1.0)) * m
        top = max(float(out.max()), 1e-6)
        preview = (out / top).clamp(0, 1)[..., None].expand(-1, -1, -1, 3)
        return (out, preview)


class Krea2EditModelPatchDynamic:
    """The upstream Krea2 Edit patch with ref_boost_mask replaced by a multiplier map."""

    @classmethod
    def INPUT_TYPES(cls):
        upstream, module = _upstream()
        spec = upstream.INPUT_TYPES()
        opt = {k: v for k, v in spec["optional"].items() if k not in ("ref_boost", "ref_boost_mask")}
        for k in ("ref_position", "ref_position_a"):
            opt[k] = (list(module.REF_POSITIONS) + [LEFT], opt[k][1])
        opt["ref_weights"] = ("MASK", {"tooltip": "per-pixel multiplier map over the reference (Krea2 Ref Weights); unconnected = 1 everywhere"})
        opt["ref_weights_all"] = ("BOOLEAN", {"default": False, "tooltip": "apply the map to both references too: for the same photo given twice, one on each side"})
        return {"required": spec["required"], "optional": opt}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "krea2edit"
    DESCRIPTION = "Krea2 Edit source patch whose reference pull is set per pixel by a weight map instead of one boost inside a 0.5-thresholded mask."

    def patch(self, model, source_latent, ref_weights=None, ref_weights_all=False, **kw):
        upstream, module = _upstream()
        _install(module)
        if ref_weights is None:
            return upstream().patch(model, source_latent, **kw)
        w = ref_weights.clone().float()
        setattr(w, _TAG, True)
        setattr(w, _ALL, bool(ref_weights_all))
        # any boost other than 1.0 makes the upstream forward ask for a bias, which our dispatcher builds
        return upstream().patch(model, source_latent, ref_boost=1.000001, ref_boost_mask=w, **kw)


NODE_CLASS_MAPPINGS = {
    "Krea2RefWeights": Krea2RefWeights,
    "Krea2EditModelPatchDynamic": Krea2EditModelPatchDynamic,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2RefWeights": "Krea2 Ref Weights",
    "Krea2EditModelPatchDynamic": "Krea2 Edit Patch (Dynamic Attention)",
}
