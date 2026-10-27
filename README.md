# LLM Explainer

Interactive chapters at `apravesh/llm-explainer`, deployed using GitHub Pages (`main / root`).

1. Text transformers (GPT2): links to Polo Club's independent Transformer Explainer.
2. Vision encoders (Pixtral – Multimodal LLMs): this project's visual explorer.
3. Mixture of Experts: an untrained minimal sparse FFN with top-1 / top-2 routing.

## Vision: real dimensions, synthetic weights

The vision chapter now **executes the full dimensions**, rather than relabeling small tensors. Pixtral is the default; ViT-B/16 remains selectable.

| | Pixtral-12B vision path | ViT-B/16 |
|---|---:|---:|
| Patch | 16 × 16 RGB (768 inputs) | 16 × 16 RGB (768 inputs) |
| Hidden features | 1,024 | 768 |
| Attention heads | 16 × 64 | 12 × 64 |
| FFN intermediate features | 4,096 | 3,072 |
| Encoder blocks | 24 | 12 |
| Position mechanism | 2D RoPE in every attention block | Learned absolute vectors at input |
| Readout | All patches → projector 1,024 → 5,120 → 5,120 | Final CLS → schematic classifier |

These are architecture dimensions, **not pretrained Pixtral/ViT inference**. Weights are dense deterministic synthetic values generated from a documented integer hash. Biases are omitted; normalization uses identity gains and zero bias. Inputs use the existing demo RGB normalization to [-1,1]. No classifier, vocabulary, tokenizer, or language decoder is executed. No model checkpoint is downloaded.

For interactive token counts, images are deliberately resized to **128 pixels on the longer side** with patch-aligned edge padding (Pixtral), or a **96 × 96 center crop** (ViT). This reduces the number of patches, not hidden widths, heads, FFN size, or depth. The default Pixtral bird image has 48 patches. Production image preprocessing/resolution and trained weights are not reproduced.

Reference dimensions: Mistral's `Pixtral-12B-2409/params.json` on Hugging Face; projector structure in `mistral-inference/src/mistral_inference/vision_encoder.py`; ViT-B/16 architecture in *An Image Is Worth 16×16 Words*.

## Runtime and performance

- A **Web Worker** computes every block in order. Later blocks depend on actual preceding outputs; none are skipped. Embeddings, completed blocks, final output, and projector results are streamed to the page.
- **WebGPU** is attempted first, with an independent numerical self-check. It generates a dense weight buffer and runs tiled FP32 matrix multiplication. Device/validation failures are reported; CPU retry is available.
- **WebAssembly SIMD** is the CPU fallback, with a plain worker JavaScript fallback when SIMD is unavailable. The 385-byte SIMD kernel is embedded in source and the offline build. It uses full dense dot products, not low-rank/sparse approximations.
- Full weights are materialized **one operation at a time** in the worker; UI inspectors regenerate individual weights from descriptors. The UI does not hold 400+ million boxed JavaScript weight values or millions of DOM cells.
- Activations are retained for block/patch/head inspection. Completed runs are cached with a maximum of two entries. Image/model changes terminate stale workers; cancel retains partial results without fabricating output.
- UI remains responsive during computation. A progress/status strip shows the selected backend, completion time, cancel/retry, and a CPU recomputation option.

Local benchmark on the development machine for 48 patches: approximately **1–1.2 seconds WebGPU**, **4.7 seconds Node WebAssembly SIMD**, through all 24 Pixtral blocks and both projector layers. These are measured examples, not device-independent guarantees. Browser CPU performance, device memory, and available GPU support vary.

## Matrices that remain readable

The overview and expanded stages retain full tensor extents and correct dimensions. Dense activations are painted on canvas. When several features share a pixel, their mean is shown and labeled **pooled overview**. Large weight previews are explicitly **sampled weight overviews**. These display summaries never change the underlying computation.

Exact numeric inspectors show **32 features at a time**, labeled e.g. `features 5088–5119 of 5,120`. Previous/next controls and direct index entry (press Enter) reach every feature. Hover gives the value at the indicated actual coordinate. Attention heads are displayed as a **stack with one selected head at the front**, rather than a grid of 16 matrices. Previous/next buttons and compact numbered selectors reach all 16 Pixtral / 12 ViT heads. Arrow keys wrap between heads; Home/End jump to the first/last. The selected head is shown as one connected Q/K/V calculation on the front card, rather than a separate preview followed by a dashboard. The brief card-change animation respects reduced motion. All heads still compute in parallel; stacking changes only the presentation. Each retains its full 64-feature dimension.

Expanded stages use the same heading toggle: **+** opens the stage and **−** closes it. There is no second stage-level Collapse button; Escape also returns to the overview.

The expanded matrix views keep a complete selectable patch rail. Matrix clicks and keyboard arrows select tokens and features. MLP diagrams connect corresponding token rows and explicitly show shared weights. The Projector keeps input, intermediate, activated, and output matrices; ViT instead shows Classification.

### Focused attention window

Expanded Self-attention shows **six query rows and six key columns at a time** around a magnified **6 × 6 attention-weight grid**. Q/K/V retain all real head features; the output preview displays eight real feature columns at a time. On desktop, the patch list starts level with the attention grid instead of the section heading; this alignment adapts to layout changes. On mobile the horizontal strip stays above the diagram. Scrolling the complete left patch pane pans the inspection window; it does not change the selected query, rerun the encoder, or rebuild the pane. Previous/next-window controls, a full-matrix minimap, and “Follow query” provide alternative navigation. A few faint context cells and explicit omitted-token counts represent the parts outside the magnified view; the minimap shades every region outside the current query/key crop.

Click an attention cell (or **Attention +**) to unfold that same six-token view into **Dot product → Scaling · Mask → Softmax**. Three panels spread apart with a short animation; Replay repeats it and Collapse returns to the simple flow. The dot-product panel sums all 64 head features, scaling divides by √64 = 8, and softmax normalizes across the complete token row. Each panel has its own labeled color range and a stage-specific value tooltip. The selected query/key is highlighted across all three panels, with its numeric formula below.

The mask defaults to **none**, because this is a vision encoder. **Compare decoder mask** is optional: future-key scores become −∞, their weights become 0, and the displayed head output/value-mixing loop uses the corresponding comparison weights. This never changes the encoder tensors, subsequent blocks, or projector results. Comparison banners make the boundary explicit. Collapse restores the unmasked encoder view, and changing models clears comparison mode. Float64 inspected dot products may differ slightly from the stored FP32 scaling/softmax values; the selected arithmetic discloses this.

Query and Key have **independent scroll windows**: scroll over a pane or drag its scrollbar, with keyboard arrows / Page Up / Page Down / Home / End available. Query moves attention rows and the matching output rows; Key moves attention columns and the matching V rows. The main patch rail and paired paging arrows can still move both windows together. The minimap shows the rectangular query/key crop, including off-diagonal windows.

The mixing operation is labeled **Attention_weights × V**, with weights identified as softmax probabilities. Its result is shown immediately to its right as a **6-row × 8-feature output snippet**. Vertical output scrolling stays linked to Query; horizontal scrolling reaches all 64 head features. The bottom labels distinguish the visible snippet from the full `N × 64` head output. Hovering a snippet cell shows the actual stored value. These are full weighted sums over all keys, never sums over only the visible key window.

This is a display crop, **not local/windowed attention**. Each circle retains its original full-row softmax probability. The crop is never re-normalized. A visible-mass note reports the selected query's attention assigned to shown keys vs. off-window keys. The output strips and automatic value-mixing loop still include **every token**. Grey context means out of view, not a causal mask. ViT includes CLS as its own token; Pixtral has no CLS.

Hovering or keyboard-focusing an attention cell immediately shows its query/key labels, percentage, and full-precision stored weight in a tooltip; the value comes from the full attention row, not a renormalized crop. Hovering does not select a different pair. The tooltip clears when the window scrolls, focus leaves, or the view changes. Selecting a cell or Q/K/V row updates the same query/key pair and exact optional inspectors. Switching head/block retains both viewed patch ranges; selecting a query or key outside its window recenters only that axis. Scroll rendering updates only the focused diagram so the patch pane and value-mixing loop do not jump or restart. The matrix retains all token rows and head features in memory. The rest of the architecture's matrices are unchanged.

Projection titles open full-dimensional multiplication overlays. Numerical feature tables, selected-score/softmax arithmetic, position details, residual output, and decoder-mask comparison stay collapsed by default. Q/K displayed here already include Pixtral's rotary positions; the projection overlay is explicitly pre-RoPE. The Q and K wires meet at the grid; V travels separately to the mixing operation, and the outputs continue on the right.

Value mixing **loops automatically** over all keys while the attention view is open; there is no play button. It pauses for hidden tabs/dialogs and respects reduced motion. It never changes the selected query/key or weights. The numeric output is explicitly a partial sum until the loop completes, and repeated frames are not announced by screen readers.

### Readable patch rows

Activation matrices retain their full dimensions, but now draw a horizontal guide for every token row (with a stronger guide every fourth row). A side ruler shows patch labels and always identifies the selected patch. Labels are spaced for legibility; the matrix still contains every row. ViT separately labels CLS and reports “36 patches + CLS.” Weight matrices are not labeled as patches: their rows are input features. Very dense token grids thin only visual guides, which the caption discloses.

### Move through all transformer blocks

Use the **small previous/next arrows on the transformer block itself** to browse Pixtral’s 24 blocks or ViT-B/16’s 12 blocks. The same inline arrows appear alongside the heading when Self-attention or MLP is expanded. There is no separate “Encoder Depth” bar or top block slider.

Self-attention and MLP update together with a short transition. The selected patch, head, feature ranges, and expanded attention/MLP view are preserved. Arrows disable at the first/last block or when the neighboring block has not finished computing. Keyboard arrows and Home/End also work while an inline block control is focused. Reduced-motion preferences skip the transition. Final encoder Output/Projector tensors remain the final result, not the inspected intermediate block.

Click either compact **MLP input or activated matrix** to expand the inspected MLP block, without changing the selected patch or transformer block. Enter/Space work too. The two small multiplication buttons between those matrices still open their own arithmetic overlays.

### Multiplication overlays

Click an **existing curved connection from a patch to its embedding** to open that patch’s projection overlay directly, without expanding a stage. Hovering or focusing any connection brightens the entire group, making the projection step feel like one clickable area; there is no extra ×W button. Transparent hit areas split the space between neighboring curves, making thin lines easier to click without changing their appearance or letting one connection cover another. Arrow keys browse the visible connections; Enter/Space opens the focused patch’s math. Connections appear once projection results are ready and never represent CLS as a projected patch. The shortcuts inside Patches/Embedding remain available.

Click patch projection, Q/K/V projection, MLP Expansion/Gate/Contraction, attention output projection, or either projector matrix to inspect **X × W = Y** with all actual dimensions. The complete output matrix stays present. Select a row/column, pause/play, step a term, scrub included terms, inspect exact weights, or page the selected output vector.

The runtime tensors use Float32. For numerical inspection, dot products accumulate the same weights in Float64. The UI explicitly reports their difference (rounding / accumulation order) rather than claiming bitwise equality. All terms are included; animation ticks may include a labeled group of terms. Multiplication playback uses a fixed **fast 2.4-second** duration for a full dot product. There is no speed selector; pause, step, replay, and the term slider remain available. Time-based playback reveals a larger prefix of the exact same sum; it never omits terms. Reduced motion shows completed results and retains manual inspection. Background tabs pause animations.

### Projector GELU overlay

Click **GELU → Explore** between the projector’s linear layers. The overlay keeps the full pre/post activation matrices visible, highlights the selected patch and feature, and plots its scalar transformation with a replayable trace. Patch/feature inputs and matrix clicks inspect the real stored FP32 tensors; a nearby-feature cutaway compares exact before/after values.

The explanation distinguishes exact GELU (`x Φ(x)`) from the runtime’s tanh approximation, which is shared by the compute engine and inspector. A separate slider and example buttons explore positive, zero, and negative inputs without changing any model values. GELU is elementwise: dimensions and patch identities remain unchanged. Escape/close restores focus to the trigger; hidden tabs and reduced-motion preferences suppress animation.

### Walkthrough and positional lab

**Follow a patch** plays through seven stages with pause/resume, previous/next, stage buttons, and replay. It uses cached real-size results; the schematic classifier/decoder boundary is disclosed. Opening another stage or changing the input stops the tour. The separate positional lab uses **four fixed toy features** (two pairs), so every term is visible. Its 30°/15° angular increments are teaching choices, not the encoder’s frequency schedule. The compact main lab shows **one query patch in one circle**, with blue row-pair and purple column-pair arrows (two separate feature pairs, not sequential rotations). The main rotation loops automatically while its circle is in view: **2.6 seconds to rotate, hold the complete result for 2 seconds, then restart**. Row/column edits restart the loop. There are no main-page playback buttons or scrubber; manual playback controls remain only in the math overlay. Scrolling away, hiding the tab, or opening an overlay pauses the loop and preserves the remaining rotation/hold time. Reduced motion shows the final result without looping. Positions are no longer capped at 7: the old bound came from a toy 8×8 grid, not RoPE. Inputs accept non-negative safe integers, and full revolutions are omitted only from the arc drawing to keep large indices cheap to render; the numerical angles and endpoints retain the actual calculation. Pair readouts separate full turns from the remaining angle. A compact disclosure explains why 361° and 1° produce identical final vectors, why rotation does not encode the turn count, and why angle periodicity must not be confused with a universal period in position indices. It explicitly notes that positions 1 and 361 happen to collide with this toy’s 30°/15° speeds. Row/column inputs remain available on the main page; scrubbing is available in the math overlay. **See the math** opens an overlay combining numerical 2×2 rotation matrices with the four-term attention dot product. **Why move two patches together?** opens a separate Q/K comparison overlay with shared-position shifts and a move-only-K contrast. This overlay animates once when opened and stays on the completed result, with no loop, timing captions, playback buttons, or scrubber. Closing and reopening replays from the start. Coordinate edits and move buttons each animate their new calculation once; move buttons remain enabled after completion. Visibility changes resume an unfinished animation but never restart a completed one. Per-pair explanations distinguish the shared frequency (degrees per row/column), the position-dependent added angle, and the vectors’ different starting directions. Each overlay copies the main example into isolated state; closing preserves the main view and restores focus. This does not change the real encoder’s 64-feature heads.

## MoE chapter

The MoE chapter is intentionally still a small teaching model, separate from the full-size vision engine: eight input features, four 8 → 16 → 8 GELU experts, an 8 → 4 router, top-1/top-2 routing, shared input LayerNorm, and a residual connection. Only selected experts run. The UI exposes gates, exact weighted mixtures, a dense comparison, load counts, and animations. It is not pretrained language-model inference.

## Run, test, build

No runtime dependencies or API keys. Node.js 20+ for local tooling:

```sh
node server.mjs
node --no-maglev --test tests/*.test.mjs
node build.mjs
```

The server binds only to loopback on port 4173 (`PORT` overrides it). `vision.html` loads `src/full-app.mjs`. `standalone.html` is the fully offline, double-clickable vision chapter, including sample SVGs, all scripts, the identical worker, and the SIMD kernel. `build.mjs` builds from the vision page, never the homepage. The `--no-maglev` test flag avoids a concurrent JIT/GC hang observed with the locally installed Node 23.7; it is not required in browsers.

Tests include the independent reference-model numerical tests, SIMD kernels vs Float64 dot products, normalization/RoPE/attention, dense dependency ordering, cancellation, exact feature windows, bounded canvas summaries, offline syntax, and a **real 48-patch, 24-block, 1,024-wide Pixtral integration run including the 5,120-wide projector**.

## Active vision implementation

- `src/full-config.mjs`: canonical dimensions, weight hash/descriptors, matrices and views.
- `src/full-backend.mjs`: validated WebGPU and CPU/SIMD matrix multiplication.
- `src/wasm-kernel.mjs`, `runtime/matmul.wat`, `runtime/matmul.wasm`: embedded SIMD kernel and auditable source/binary.
- `src/full-engine.mjs`, `src/full-worker.mjs`: full-size dense forward pass and streamed results.
- `src/full-matrix.mjs`, `src/matrix-row-guides.mjs`: bounded canvas overviews, patch row guides/rulers, exact coordinate values and feature paging.
- `src/full-operations.mjs`, `src/full-overlay.mjs`: inspectable operations and animated dot products.
- `src/block-navigation.mjs`: inline block-arrow navigation and cancellable, reduced-motion-aware transitions. Timing/handoff helpers remain available to reference tests but no longer add a toolbar to the page.
- `src/attention-window.mjs`: six-token display window, minimap, exact full-matrix weights, and disposable patch-scroll observation.
- `src/head-stack.mjs`, `src/full-views.mjs`, `src/full-app.mjs`: stacked-head presentation, model UI, matrix flows, patch selection, loading lifecycle and tour.
- `src/motion.mjs`: in-place stage zoom with native transitions and fallback.

The earlier `engine.mjs`, `app.mjs`, and related small-model view modules remain as reference fixtures for numerical/regression tests. **They are not imported by the active vision entry point or included in the offline application.** Their 24-feature tensors are not used as stand-ins for real dimensions.

## Deployment and attribution

GitHub Pages serves `main / root`, with `.nojekyll`, no external hosting. `index.html` and the MoE chapter remain separate. Rebuild the offline file after edits, run tests, and publish all new full-runtime modules together with HTML/styles; then verify the Pages deployment.

This is an original independent explainer inspired by the interaction style of Polo Club's Transformer Explainer. It does not copy their code/assets and is not affiliated with Georgia Tech, Polo Club, or Mistral. Sample illustrations are original SVG artwork. Images stay on the user's device; there are no analytics or inference services.
