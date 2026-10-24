# ComfyUI Krea2 Edit — Dynamic Attention

Builds on [comfyui-krea2edit](https://github.com/lbouaraba/comfyui-krea2edit) (≥ 2.0.0, Apache-2.0), which must be installed. No upstream code is copied; its patch node is called and only the reference attention bias is replaced.

The upstream `ref_boost_mask` is thresholded at 0.5, so a reference gets two strengths at most: the boost inside the mask and 1 outside. Here every pixel of the reference carries its own multiplier on target→reference attention (added to the logits as its log), so the face, hair, skin, clothes and background can each pull as hard as you choose.

## Nodes

- **Krea2 Ref Weights** — a base multiplier plus up to six masks, each laid over the map with its own weight (later masks on top, soft edges kept). Outputs the map and a preview.
- **Krea2 Edit Patch (Dynamic Attention)** — the upstream source patch with every input except `ref_boost` / `ref_boost_mask`, plus `ref_weights`. Unconnected, it is the upstream patch at boost 1.

1 = as trained, 2 = pulls twice as hard, 0.05 = nearly ignored.
