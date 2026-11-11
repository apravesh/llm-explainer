# LLM Explainer

Interactive chapters at `apravesh/llm-explainer`, deployed using GitHub Pages (`main / root`).

1. Text transformers (GPT2): links to Polo Club's independent Transformer Explainer.
2. Vision encoders (Pixtral – Multimodal LLMs): this project's visual explorer.
3. Mixture of Experts: an untrained minimal sparse FFN with top-1 / top-2 routing.
4. Parameters & compute: an interactive architecture ledger and GPU-mechanics primer.

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
- UI remains responsive during computation. A progress/status strip shows the selected backend, completion time, cancel, and a CPU retry option shown only after a failure or cancellation.

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

Click either compact **MLP input or activated matrix** to expand the inspected MLP block, without changing the selected patch or transformer block. Enter/Space work too. The small multiplication buttons between those matrices open the same unified, in-place MLP calculation. In expanded MLP, the detailed numeric tables sit behind a closed-by-default **Show actual values** disclosure. Opening it reveals the same exact feature inspectors; the choice stays open when inspecting another patch, feature, or block. The connected matrices, dimensions, and projection buttons remain visible.

### Multiplication overlays

Click an **existing curved connection from a patch to its embedding** to open that patch’s projection overlay directly, without expanding a stage. Hovering or focusing any connection brightens the entire group, making the projection step feel like one clickable area; there is no extra ×W button. Transparent hit areas split the space between neighboring curves, making thin lines easier to click without changing their appearance or letting one connection cover another. Arrow keys browse the visible connections; Enter/Space opens the focused patch’s math. Connections appear once projection results are ready and never represent CLS as a projected patch. The shortcuts inside Patches/Embedding remain available.

Click patch projection, Q/K/V projection, attention output projection, or either projector matrix to inspect **X × W = Y** with all actual dimensions. The complete output matrix stays present. Select a row/column, pause/play, step a term, scrub included terms, inspect exact weights, or page the selected output vector.

The runtime tensors use Float32. For numerical inspection, dot products accumulate the same weights in Float64. The UI explicitly reports their difference (rounding / accumulation order) rather than claiming bitwise equality. All terms are included; animation ticks may include a labeled group of terms. Multiplication playback uses a fixed **fast 2.4-second** duration for a full dot product. There is no speed selector; pause, step, replay, and the term slider remain available. Time-based playback reveals a larger prefix of the exact same sum; it never omits terms. Reduced motion shows completed results and retains manual inspection. Background tabs pause animations.

### Arrow-based MLP flow

The main expanded MLP uses the same branch-and-merge layout as the SwiGLU lesson, but reads the **actual selected token and block**, not the toy MLP. One normalized x branches into W₁ → SiLU and W₃; the product feeds W₂, then the resulting update Δ is added to the separate pre-normalization residual H. ViT instead shows W₁ → GELU → W₂ with no fictitious gate. Cards label the full 1,024/4,096 (or 768/3,072) dimensions. Colored strips pool adjacent features into at most 16 display cells and scale their colors independently; exact numeric values remain available in the collapsed disclosure. No model tensors or weights are changed.

A short trace highlights the parallel branches, their product, contraction and residual in order. The diagram remains visible while the optional full calculation opens beneath it. Patch/block selections refresh the same flow; replay and keyboard-operable projection cards remain available. The actual-values disclosure stays closed by default. The arrow animation is finite and respects reduced motion and hidden tabs.

### Unified MLP calculation

W₁, W₃ and W₂ cards and **Inspect full calculation** all unfold **one in-place calculation beneath the persistent arrow flow** inside the MLP stage, not three separate dialogs. Pixtral shows `X × W₁ = A` and `X × W₃ = B` side by side, `C = SiLU(A) ⊙ B`, then `C × W₂ = Δ` and the residual result `H′ = H + Δ`. ViT shows the two linear operations with GELU and no invented W₃ branch. Complete full-sized input, weight, product and residual matrices remain visible alongside the existing patch rail and block navigation.

Replay traces a selected dot product in both expansion branches simultaneously, then activation/gating, then the contraction. Each dot product includes every input feature. The displayed product matrices are cached complete tensors, not partially computed animation values. Pause/resume is available at a fixed timing; actual prefix sums, term scrubbers, FP32 outputs, and rounding differences remain in **Show actual values**. Selecting another patch, block, or feature preserves the calculation layout and reveals its completed result without replaying. Escape or **− Calculation** returns to the arrow-based MLP flow without collapsing the whole stage. Reduced motion shows complete results; hidden tabs pause playback.

### Projector GELU overlay

Click **GELU → Explore** between the projector’s linear layers. The overlay keeps the full pre/post activation matrices visible, highlights the selected patch and feature, and plots its scalar transformation with a replayable trace. Patch/feature inputs and matrix clicks inspect the real stored FP32 tensors; a nearby-feature cutaway compares exact before/after values.

The explanation distinguishes exact GELU (`x Φ(x)`) from the runtime’s tanh approximation, which is shared by the compute engine and inspector. A separate slider and example buttons explore positive, zero, and negative inputs without changing any model values. GELU is elementwise: dimensions and patch identities remain unchanged. Escape/close restores focus to the trigger; hidden tabs and reduced-motion preferences suppress animation.

### Positional lab

The separate positional lab uses **four fixed toy features** (two pairs), so every term is visible. Its 30°/15° angular increments are teaching choices, not the encoder’s frequency schedule. The compact main lab shows **one query patch in one circle**, with blue row-pair and purple column-pair arrows (two separate feature pairs, not sequential rotations). The main rotation loops automatically while its circle is in view: **2.6 seconds to rotate, hold the complete result for 2 seconds, then restart**. Row/column edits restart the loop. There are no main-page playback buttons or scrubber; manual playback controls remain only in the math overlay. Scrolling away, hiding the tab, or opening an overlay pauses the loop and preserves the remaining rotation/hold time. Reduced motion shows the final result without looping. Positions are no longer capped at 7: the old bound came from a toy 8×8 grid, not RoPE. Inputs accept non-negative safe integers, and full revolutions are omitted only from the arc drawing to keep large indices cheap to render; the numerical angles and endpoints retain the actual calculation. Pair readouts separate full turns from the remaining angle. A compact disclosure explains why 361° and 1° produce identical final vectors, why rotation does not encode the turn count, and why angle periodicity must not be confused with a universal period in position indices. It explicitly notes that positions 1 and 361 happen to collide with this toy’s 30°/15° speeds. Row/column inputs remain available on the main page; scrubbing is available in the math overlay. **See the math** opens an overlay combining numerical 2×2 rotation matrices with the four-term attention dot product. **Why move two patches together?** opens a separate Q/K comparison overlay with shared-position shifts and a move-only-K contrast. This overlay animates once when opened and stays on the completed result, with no loop, timing captions, playback buttons, or scrubber. Closing and reopening replays from the start. Coordinate edits and move buttons each animate their new calculation once; move buttons remain enabled after completion. Visibility changes resume an unfinished animation but never restart a completed one. Per-pair explanations distinguish the shared frequency (degrees per row/column), the position-dependent added angle, and the vectors’ different starting directions. Each overlay copies the main example into isolated state; closing preserves the main view and restores focus. This does not change the real encoder’s 64-feature heads.

### SwiGLU lab

Immediately below 2D RoPE, the SwiGLU section contrasts GPT-2’s single-expansion GELU MLP with two parallel projections and a matching-feature product. The diagram uses Mistral’s `SiLU(xW₁) ⊙ (xW₃)` followed by `W₂`; W₃ is labeled a parallel expansion, not a sequential gate layer.

A separate deterministic 4 → 6 → 4 toy MLP lets readers switch between three input patches with the same fixed weights. Its dependency trace highlights both branches together, then their elementwise product and the contraction. It plays once on entry, supports replay, pauses off-screen/when hidden, and respects reduced motion. All vectors remain visible throughout; actual values are behind a closed-by-default disclosure that remains open when changing toy patches. Pixtral’s real 1,024/4,096 dimensions, the residual boundary, and the synthetic nature of the example are explicit. It does not alter the encoder.

## MoE chapter

The MoE chapter is a four-block, eight-feature teaching transformer, separate from the full-size vision engine. Each block has two four-feature causal attention heads, its own Q/K/V/Wₒ matrices, an independent 8 → 4 router and four 8 → 16 → 8 experts (GELU by default, optionally SwiGLU). Top-2 is the default; top-1 and a dense FFN comparison remain selectable. Only selected experts run. LayerNorm precedes attention and the FFN; each has its own residual addition. Seeded token vectors receive sinusoidal positions once, and final LayerNorm plus an 8 × 8 symbolic vocabulary head follows all blocks. This is not pretrained language-model inference.

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
- `src/head-stack.mjs`, `src/full-views.mjs`, `src/full-app.mjs`: stacked-head presentation, model UI, matrix flows, patch selection and loading lifecycle.
- `src/motion.mjs`: in-place stage zoom with native transitions and fallback.

The earlier `engine.mjs`, `app.mjs`, and related small-model view modules remain as reference fixtures for numerical/regression tests. **They are not imported by the active vision entry point or included in the offline application.** Their 24-feature tensors are not used as stand-ins for real dimensions.

## Deployment and attribution

GitHub Pages serves `main / root`, with `.nojekyll`, no external hosting. `index.html` and the MoE chapter remain separate. Rebuild the offline file after edits, run tests, and publish all new full-runtime modules together with HTML/styles; then verify the Pages deployment.

This is an original independent explainer inspired by the interaction style of Polo Club's Transformer Explainer. It does not copy their code/assets and is not affiliated with Georgia Tech, Polo Club, or Mistral. Sample illustrations are original SVG artwork. Images stay on the user's device; there are no analytics or inference services.

### MoE expert architecture and parameter budgets

The local MoE chapter keeps routing separate from the expert activation choice. Selecting SwiGLU adds a separate 8 × 16 W₃ matrix per expert, applies SiLU to W₁'s output, multiplies matching intermediate features, and contracts through W₂. The router and common expert weights retain their seeds when switching architectures. Only selected experts execute in either mode. The input is sent in full to each selected expert, not split across experts.

The page makes routing probabilities versus normalized mixture weights explicit. GELU expert weights total 256 per expert; SwiGLU totals 384. Four experts plus the 32-weight router therefore store 1,056 or 1,568 parameters. Top-1 uses 288 or 416; top-2 uses 544 or 800. The cost table and bars show stored parameters, active parameters per token, and linear MACs, with the exclusions and non-speedup caveat stated. A collapsed parameter-budget explanation shows how standard attention's 4d² compares with the 2df/3df MLP formulas; each block actually executes the 256-weight attention projections plus causal head mixing. The FFN comparison table remains per-block and excludes attention and the final head.

Expert inspection uses a process diagram before optional numerical multiplication tables. Internal SiLU feature gates are explicitly distinguished from router weights. Numeric disclosures remain open/closed across token and architecture changes. The redundant top-of-page token-tour shortcut is removed; the in-flow animation automatically loops through the selected token’s cached routing, expert computation, mixture, and residual, holding the completed result before restarting. There are no stop/replay controls. It pauses off-screen, while the tab is hidden, or while the method dialog is open; reduced motion shows the complete result without looping.

## Parameters & compute chapter

`analytics.html` is a separate analytical chapter, linked from the homepage and bottom of Vision/MoE. It compares ordinary MLP, same-width SwiGLU, budget-matched SwiGLU (rounded up to 64), and routed SwiGLU MoE, with MHA/GQA/MQA projections. `src/architecture-budget.mjs` validates inputs and computes exact bias-free weight counts and explicitly approximate forward FLOPs, weight storage, decoder KV cache, a naïve score buffer, and an isolated GEMM roofline lower bound. None of these estimates execute a model or predict measured GPU latency. The page cites primary papers and NVIDIA documentation and discloses omitted embeddings/projectors/norms, runtime overhead, causal-vs-bidirectional work, optimizer state and expert communication.

### Vector-first MoE view

The MoE overview represents each token state, expert output, weighted contribution and residual result as an eight-cell horizontal vector, with exact values on hover. Solid arrows connect the full normalized x directly to selected expert inputs; the router's scalar weights modulate outputs rather than replacing x. The same feature ordering is preserved in raw and weighted strips. Inactive experts show no fabricated activations. Input editors and combine/residual numeric tables are opt-in details. On narrow screens the connected diagram scrolls horizontally instead of dropping the arrows or turning the computation into disconnected cards.

The analytics chapter has three accessible visual tabs: Parameters (a transformer block with per-projection allocations and selectable MLP/SwiGLU/MoE design), Token compute & memory (per-block FLOP flow and separately scoped weight/cache/score allocations for the MoE comparison), and GPU execution (device memory → tile reuse → matrix operations with isolated GEMM resource lower bounds). Shared configuration survives tab changes. Tables and derivations are opt-in below the diagrams; colors/meters do not imply physical GPU layout or measured latency.

The two compact, non-expandable sections inside the MoE block now show genuinely computed causal attention and attention projection Wₒ plus residual. They animate before the routed FFN. A single shared previous/next block control inspects cached results without rerunning the stack, preserving the chosen token and open inspector. The final readout is outside the block enclosure and always uses block 4, even when block 1 is being inspected. It labels v0–v7 as untrained symbolic vocabulary slots; no actual words are predicted or generated.

### Computed MoE block stack

`src/moe-transformer.mjs` defines four deterministic independent blocks, two causal heads, initial sinusoidal position vectors and a final untrained vocabulary head. It keeps attention input/output, head results, Wₒ update, per-token FFN traces and logits inspectable. `src/moe-block-view.mjs` renders the compact attention and final readout vectors. The numerical pipeline is H → Norm → attention → Wₒ → +H → h → Norm → router/experts → +h; the result is the next block's H. Input edits target the original seeded vector before positional addition and recompute the entire stack. GELU/SwiGLU changes preserve attention/router/common matrix seeds. Dense mode applies expert 1 within each block, not the same expert weights across every block. Fixed positions and non-affine normalization have no learned parameters. Four block parameter budgets plus 64 LM-head weights cover the toy's learned matrices; no learned embedding table, RoPE, tokenizer or text sampling is simulated.

The MoE block outline is a light dotted boundary, with its single block navigator embedded in the top edge. Attention Wₒ is a compact non-interactive arrow labeled “× Wₒ” and “+ H”; the underlying projection and residual are still computed. At laptop widths (1024–1440px verified), the complete block fits without horizontal scrolling; the token list moves above it on smaller laptops. Narrow phone screens retain horizontal scrolling rather than shrinking the vector math to illegibility. The final norm and LM head remain outside the repeated-block boundary.

## Training chapters

The homepage groups existing text, vision, MoE and parameter chapters under **Architecture**. **Training** has `pretraining.html`, `post-training-instruct.html` and `post-training-rl.html`. All three Training pages are interactive chapters. All links preserve the GitHub Pages project prefix.

The pretraining page keeps three short visual lessons: (1) a seven-record hand-authored data pipeline illustrating extraction/filtering, exact deduplication, a document-level holdout and storage shards; (2) a character-level BPE tokenizer with frequency-weighted adjacent-pair counts, ordered merges and encoding into vocabulary IDs; (3) a truly trainable tiny causal transformer with width 4, one head, tanh FFN 4→8→4, fixed positions, non-affine LayerNorm, residuals and an untied eight-token LM head (192 weights). It computes shifted next-token targets, stable softmax cross-entropy averaged across valid batch tokens, scalar-autodiff gradients and SGD at learning rate 0.1. Manual steps or a finite automatic sequence expose forward/loss/backward/update stages; reset cancels stale callbacks. The page keeps its pre-update forward snapshot while comparing the same-batch loss after a real update.

All examples are local and deliberately isolated: no crawl is performed, the small BPE vocabulary does not feed the neural demo, and word IDs in the neural demo are fixed for legibility. Repeatedly fitting the displayed training batch is not validation or language-model capability. The RL chapter uses a computed categorical policy, not a real LLM. Sources are linked, and longer caveats live in disclosures.

### Instruction fine-tuning chapter

`post-training-instruct.html` covers a fictional chat template with clickable special-token IDs, training versus generation prefixes, a reviewed demonstration-data pipeline, and a two-sequence assistant-only versus full-target loss comparison. All example records and probabilities are illustrative. `src/instruct-engine.mjs` computes normalized probability vectors, masked cross-entropy and vocabulary-logit derivatives; it does not run a model or update weights. The special tokens/IDs are deliberately not attributed to a real checkpoint. The RL chapter follows with rewards, policy updates and checkpoint refreshes.

### GRPO and checkpoint chapter

`post-training-rl.html` shows rollout inference, group rewards/advantages, three learner updates, then an in-memory snapshot and rollout-policy refresh. `src/rl-engine.mjs` is a genuinely updated three-logit categorical policy for one-token answers to “What is 2 + 2?”: four seeded draws, population-standardized group advantages (zero for equal rewards), clipped surrogate with epsilon 0.2, SGD learning rate 0.15 and beta=0 (no KL/reference model). Rollout probabilities and advantages stay frozen for all three updates. Snapshots are independent arrays; inference uses a separately selected snapshot and independent RNG, never mutating training state. The old rollout policy, optional KL reference, learner, durable checkpoint and serving selection are distinguished in the explanation.

Manual steps or finite “Play one round” playback drive the lesson. Pause/reset, reward-rule changes, hidden tabs and reduced-motion changes cancel pending callbacks. The constant-reward option illustrates zero relative signal. Saved snapshots exist in browser memory only. Changing their arithmetic accuracy is not evidence of real-world capability. The chapter does not download models, save checkpoint files or call inference services.

### Pretraining visuals

The three pretraining labs are animation-first. **Data** shows the selected page as one raw HTML string, strikes out and collapses markup and page chrome (`htmlSegments`), drops spam/unapproved/duplicate pages, then joins the survivors into one separator-delimited corpus string with a shard cut and a held-out validation string (`corpusTape`). **Tokenizer** highlights every occurrence of the winning pair, fuses the tiles, grows a colour-coded vocabulary shelf and tracks token/vocabulary meters; a free-text box replays the learned rules on a new word (`encodeSteps`). **Learn** is a five-stage pipeline: a forward pulse fills activation strips and causal attention weights (future tokens hatched), the loss node scores the guess, a backward pulse fills gradient strips, and the update shows `W − 0.1 × ∂L/∂W = W′` for the LM head, the probability bars moving, and a loss curve. "Train 30 updates" repeats real SGD steps. All animations reveal computed values only, pause in hidden tabs and respect reduced motion.

### Post-training visuals

**Instruct.** Chat bubbles are serialized into one template string and then token chips (staggered reveal, replayable; generation mode ends in a caret and an empty "model writes here" slot). Demonstration records are small chats that get stamped accepted/rejected/held out, and the surviving train record is shown with its graded tokens. The loss-mask lab draws every next-token guess as a tile with a loss bar; selecting a graded tile sweeps a gradient underline back through all the context it could see, and a −ln p curve follows the probability slider. Probabilities stay illustrative; no model trains on this page.

**RL.** The frozen rollout copy shows its probabilities as a strip, and each sampled answer is a pin dropped at its uniform random number (`random` from `rl-engine.mjs`, the same draws `sampleGroup` makes). Answers are cards that get a reward stamp and an advantage bar above or below the group-average line. The learner panel keeps a tick at the frozen copy's probabilities and plots p("4") after every update; "Run 5 rounds" repeats the real loop. Snapshots are a clickable bar timeline, and inference draws one pin (or 20) with a running tally against the expected probabilities.

### Shared header

Every page uses the same `.site-bar` header from `site-header.css`: the `[ ] LLM EXPLAINER` brand on the left and one chapter list (All chapters · Architecture: Vision, MoE, Parameters · Training: Pretraining, Instruct, RL) with the current page marked. Page-specific controls (Vision's About, MoE's Method & limitations, in-page jump links) sit in a `.page-links` row under the page intro instead of in the header. `build.mjs` inlines `site-header.css` into `standalone.html`.

### One visual language

The home page, Parameters & compute and the three Training pages use the Vision/MoE look: cool `#f8f9fb` page, `#232e3a` ink, `#3667e4` blue accent, sans headings with a serif-italic blue emphasis line, and small-caps sans eyebrows. Green, coral and purple remain only as category or meaning colours (MLP vs attention, correct vs wrong, user vs assistant text, gradients).

### Shared column and title type

`site-header.css` also carries the page layout shared by every chapter. Text sits in Vision's reading column (gutter `clamp(16px, 4.6vw, 64px)`, at most 1280px wide), and the header brand lines up with it. Interactive labs break out to the full page width as white bands, like the Vision and MoE explorers. Page titles, eyebrows, intro paragraphs and section headings use MoE's type: weight 500, −1.3px tracking, with the emphasis line in serif italic `#6681b5`.

The Parameters chapter no longer has the old "01 · Where do the parameters go?" block diagram and comparison table; the formulas-first section replaces it. The shared dimension controls still drive the Design choices, Token compute & memory and GPU execution tabs.

The formulas-first section has a **Real model** picker (`PF_PRESETS` in `src/param-formulas.mjs`): GPT-2 small, Mistral 7B, Llama 3 8B, Mixtral 8x7B and the Pixtral 12B vision encoder, using published dimensions. The engine now models GELU (2·d·f) versus SwiGLU (3·d·f) MLPs, an explicit f, grouped-query attention (2d² + 2·d·d_kv) and learned position tables, so the bias-free totals land within about 1% of the official parameter counts.

The RL loop lab opens with an overview of the whole loop (prompt → frozen copy runs 4× → responses → reward function → group average → update learner ×3, with a dashed return path for the weight copy); the stage in progress is highlighted. Below it, each of the four rollouts has its own lane: prompt in, the frozen model's probability strip with that run's random pin, the response, its reward, its advantage against the group average, and the learning signal with the current probability ratio. The lab spans the full page width, with the frozen and learner copies beside the lanes.

The RL page's loop lab has three method tabs. **REINFORCE** is the default: one model, a mean baseline (`meanBaseline`), one gradient step per group (`reinforceUpdate`, learning rate 1). **GRPO** keeps the earlier frozen-copy, ratio and clipping loop. **DPO** has no sampling or reward: two preference pairs for the same prompt are scored against a frozen reference (`dpoObjective`, β = 1), and each update widens the margin. All three share the overview row, the lanes, the model panels and the snapshot timeline; each has its own collapsed math section.

To make the REINFORCE/GRPO difference visible, a banner above the lanes counts how often the current answers have been used (one pip for REINFORCE, three for GRPO), and each GRPO lane has a drift meter: the learner's probability ratio against the frozen copy, the ±20% band, and a purple "paused" state once the favoured limit is passed.
