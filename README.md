# LLM Explainer

A growing collection of interactive explanations of language and multimodal model architectures.

Repository: `apravesh/llm-explainer`.

**Website:** https://apravesh.github.io/llm-explainer/

The repository is public. GitHub Pages publishes the root of `main` over HTTPS; no external host or custom deployment workflow is required.

## Current chapter: Vision Encoder Explainer

The first chapter explains how image patches become contextual representations in ViT and Pixtral, then connect to a text decoder. It is an independent companion inspired by [Transformer Explainer](https://poloclub.github.io/transformer-explainer/).

The current root page opens the vision chapter. The repository name is deliberately broader than this first chapter so more topics can be added later. The existing site title stays **Vision Explainer** to accurately identify the chapter being shown.

## Open instantly

Double-click **`standalone.html`**. It is a self-contained offline build with embedded styles, JavaScript, and sample artwork. No server or installation is required for this file.

## Run the modular source

Requires Node.js 20 or newer. No dependencies, package installation, API keys, or model downloads.

```sh
node server.mjs
```

Open `http://127.0.0.1:4173`. Set `PORT=8080` to choose another port. The server binds to loopback only.

Alternatively, serve this directory with any static HTTP server. Use an HTTP server rather than opening `index.html` as a `file://` URL, because the application uses JavaScript modules.

## Full-width diagram

The interactive architecture and its expanded stages use the full available page width, without the reading column’s width cap. The introduction and explanatory sections remain centered at a readable width. The layout uses page-width grid tracks rather than viewport-width overflow tricks, so it also fits narrow screens and pages with a scrollbar.

## Persistent patch sequence

Every expanded stage retains the **complete patch sequence** next to its computation. The left-hand column shows actual patch thumbnails, indices, and grid coordinates; scroll to reach every patch. Click a thumbnail to inspect its pixels, projection, attention, MLP, or readout without leaving the stage. On phones this becomes a horizontal strip to leave room for the numerical explanation.

Selection carries across stages, and the column retains its scroll position when inspecting a pixel or weight. Arrow keys select adjacent patches; Home/End jump to the first/last token. ViT adds a selectable [CLS] entry from Embedding onward; Pixtral has only patch entries. The thumbnails identify source patches even when the displayed computation uses deeper hidden states.

## Expand the architecture in place

Click the **+** beside any stage, or click an image patch, to unfold the computation *inside the architecture*. Neighboring stages stay visible as navigation. Click the same heading, **Collapse**, or press **Escape** to restore the overview.

- **Input:** choose a patch in the image, zoom into its individual pixels, and read RGB channel values.
- **Patches:** a literal pixel grid and a row-major RGB sequence. Select a pixel/channel to trace its flattened index, original channel value, and fixed normalization. There are no learned weights or hidden dimensions in this stage. One patch becomes one input vector, not one token per pixel value.
- **Embedding:** the flattened inputs multiplied by an inspectable slice of the actual shared projection weights, producing all 24 hidden dimensions. Then inspect the exact position addition (ViT) or input RMS normalization (Pixtral). Patch and channel selection carry over from Patches. [CLS] is treated separately, never as a projected patch.
- **Self-attention:** a connected Q/K/V → scaled scores → softmax → weighted-value-sum diagram. All three projections and both matrices stay visible. Click a node to open its precise arithmetic below the flow; choose a query from the patch column and a key from either matrix or the key strip. Pixtral visibly inserts 2D RoPE on the Q/K paths, never on V.
- **MLP:** inspect expansion, gating (Pixtral), contraction, activation, and the residual update.
- **Output:** inspect the final encoder matrix: one 24-dimensional vector per patch, plus [CLS] for ViT. This is not class logits or generated text.
- **What next?:** follow ViT’s learned [CLS] state into a classification head, or Pixtral’s patch vectors through the projector into a joint image/text decoder sequence. The projector arithmetic moved here so the vision encoder’s output is clearly separated from the language-model interface.

The displayed weight matrix is a labeled cutaway: nine input rows and four output columns. Omitted rows are marked, and **the sum always includes every input**, not only the displayed slice. Changing the selected output moves the column window. Pixel selection has arrow-key navigation and numeric row/column alternatives.

### Zoom transitions

Expansion now uses a spatial zoom rather than a simple appearance change. The stage grows into the detail view, headings move to their new positions, and the selected patch travels from its thumbnail to the pixel grid. Collapse reverses that journey. Timing is deliberately paced: 1.05 seconds to open, 0.85 seconds to close, and 0.75 seconds to switch stages. Switching expanded stages animates between them without changing the selected model or patch.

The preferred path uses native same-document View Transitions. A geometric Web Animations fallback handles browsers without that API. Reduced-motion preferences skip the animation entirely. Repeated clicks, Escape, resize, and edits settle the current transition safely instead of queueing stale state updates. Ordinary pixel/weight inspections do not replay the zoom.

## Encoder output, [CLS], and the text-decoder handoff

A seventh **“What next?”** stage sits immediately to the right of Output on desktop (below it on narrow screens). Output now ends explicitly at the encoder boundary; Next explains the task-specific readout.

- **ViT classification:** [CLS] is a trainable vector prepended before the patches, not a patch or a label. A selector shows its actual starting vector, input after positional addition, each block’s state, and final normalized state. A classifier would read that last state; no classifier is executed here. It is trained to serve as a summary in a real model, not guaranteed to summarize anything in this untrained demo.
- **Pixtral:** all patch vectors are kept, with no [CLS]. The actual teaching projector maps each 24D vector to 32D. Both matrix multiplications remain inspectable. The full-size 12B model’s corresponding widths are 1,024 → 5,120 → 5,120.
- **Decoder interface:** projected image features occupy their own input slots alongside text embeddings of the same width, with row-break and image-end tokens. Pixtral uses a causal decoder over this combined sequence rather than a separate encoder–decoder cross-attention module.
- **Symbolic generation walkthrough:** T₀ and T₁ stand for illustrative prompt slots, not actual tokenizer output. The last prefix position predicts t₁; after that token is appended, the next pass predicts t₂, and so on. These are placeholders; no tokenizer, decoder, vocabulary logits, sampling, or text generation is run.
- Switching paths updates the selected model. Clicking an image slot selects that patch and updates its actual projected vector. Generated-step controls never modify the image vectors or the encoder result.

## Multi-head attention overview

Self-attention opens with **all three teaching-model heads side by side**. Projected Q/K/V features are split into three color-coded groups of eight. Each head shows its complete N × 8 Q, K, and V matrices, an N × N attention preview, and the selected query’s eight-value output. Every head sees all tokens and projects from all 24 input features; it is not assigned a subset of patches or input features.

Click a head to reveal its existing connected attention flow below the overview, without hiding the other head previews or patch column. “Back to all heads” closes only that detail. The head dropdown and feature-group buttons select the same head.

The lower flow concatenates the three outputs **in head order** (8 + 8 + 8 = 24), applies the actual 24 × 24 output projection Wₒ, and adds the original input residual. It does not average the heads. Click Wₒ to inspect that numerical multiplication. “Animate concatenation” moves each head’s actual output into its matching slot; it respects reduced motion and cleans up on stop, resize, or navigation.

**Dimension distinction:** the feature dimension gets smaller per head. The N × N attention matrices retain the same token dimensions and are only *drawn* smaller. Attention preview intensity is normalized within each head. Full-size ViT-B/16 has 12 heads × 64 dimensions; Pixtral-ViT (12B) has 16 × 64. The executable model here intentionally remains 3 × 8.

## Connected attention and encoder/decoder comparison

Expanding a head keeps the familiar text-transformer flow in view rather than replacing it with Q/K/V tabs. Blue queries and orange keys meet at the score matrix; green values bypass the score calculation and feed the weighted sum. The selected query row and key cell are linked across the matrices.

- **Encoder is the default:** dense bidirectional attention, with a persistent “no causal mask” callout.
- **Compare with decoder masking:** the same original Q/K/V are reused, future sequence positions are blocked, and allowed weights renormalize. This is a counterfactual for all heads in the inspected attention block, **not** another model run; the merge preview uses the comparison outputs, but the actual encoder, later blocks, MLPs, and final readouts are unchanged. For ViT, the prepended [CLS] is included in that comparison ordering.
- **Animate value mixing:** each key’s weighted V vector is added to an explicitly labeled partial sum, ending at the exact full head output. Stop, stage changes, new inputs, and reduced-motion settings are handled without stale animation updates.
- **Deeper arithmetic:** click Q, K, V, the score matrix title, Softmax, or Weighted sum to inspect the corresponding calculation below the diagram. Q/K/V inspectors retain the actual projection-matrix cutaway.
- **Pixtral:** inspect before/after rotated queries and keys. The text distinguishes “no causal mask” from the block-diagonal boundary mask used for packed images.

The comparison does not train or identify semantic roles in these untrained heads. The two matrices have different color scales: signed scaled dot products versus nonnegative softmax weights. Keyboard arrows on either matrix change the selected query/key pair.

## Explore

- **Choose an image:** three original SVG illustrations, or a local raster image (PNG, JPEG, WebP, GIF, AVIF; up to 20 MB). Images never leave the device. Animated formats use a decoded frame, not video inference.
- **Follow a patch:** click image patches or sequence thumbnails, or use the seven stage headings and previous/next controls.
- **Change the architecture:** ViT uses a teaching-sized fixed square crop, input position vectors, [CLS], LayerNorm, and a GELU MLP. Pixtral uses aspect-preserving resizing/edge padding, RMSNorm, per-layer 2D RoPE, a gated FFN, and a two-layer projector.
- **Inspect attention:** switch blocks and heads; click the attention matrix to select a query/key pair. Expand the numerical inspector for Q, K, V, coordinate-wise products, the scaled dot product, and the normalized weight.
- **Change patch size:** actual RGB patchification and projection recompute with 16, 24, or 32 pixel patches.
- **Try 2D RoPE:** compare 1D/2D rotations, change coordinates and frequencies, and shift both positions to verify the relative-position identity.
- **Compare the full architectures:** the lower comparison uses ViT-B/16 and the original Pixtral 12B encoder, separately from the small teaching model.

## What is real—and what is illustrative

All displayed encoder activations, attention scores, softmax weights, FFN updates, and projector values are computed locally from the current image. There are no canned or randomly regenerated attention maps.

**The model is deliberately untrained.** Parameters are deterministic seeded random weights, not pretrained ViT/Pixtral checkpoints. It has 4 blocks, hidden width 24, 3 heads (8 dimensions per head), FFN width 48, and a 32-dimensional teaching language-model interface. There are no meaningful classifications, learned head roles, or text generation. The vision-to-language stream and packing-mask diagram illustrate the architecture; an actual LLM and multi-image batching are not executed.

The RoPE lab is a separate mathematical experiment with fixed 8D Q/K vectors, four frequency pairs, base 10,000, and adjustable frequency scale. It shows two of the four pair planes. The score sums all four pair contributions without attention scaling or softmax. It uses the paper's alternating row/column pair convention; implementations can arrange those subspaces differently.

Full-model details are sourced from:

- [An Image Is Worth 16×16 Words](https://arxiv.org/abs/2010.11929)
- [Pixtral 12B](https://arxiv.org/abs/2410.07073)
- [RoFormer](https://arxiv.org/abs/2104.09864)
- [Mistral's reference inference implementation](https://github.com/mistralai/mistral-inference)

This is an original implementation, inspired by the exploration pattern of the Polo Club's Transformer Explainer. It does not copy its source code or assets and is not affiliated with Georgia Tech, the Polo Club, or Mistral AI. Sample artwork is original SVG code included here.

## Test

```sh
node --test tests/*.test.mjs
```

The numerical suite covers deterministic parameters, pixel ordering, patch alignment, stable softmax, bidirectional attention, normalization, residual identities, shape contracts, image-sensitive results, RoPE norm preservation, independent 2D axes, relative-position invariance, and fresh RoPE in every Pixtral-style layer.

Manual browser checks cover in-place expansion/collapse, pixels and channels, output-column selection, [CLS], model/image/patch controls, head/block changes, patch and matrix selection, the numerical inspector, RoPE translation invariance, the about dialog, and responsive layouts down to 320 px.

## Files

- `index.html` — semantic page structure and educational content
- `styles.css` — responsive visual system
- `src/app.mjs` — controls, image preprocessing, visualizations, and inspection
- `src/engine.mjs` — testable, deterministic numerical model with inspectable weight and activation traces
- `src/expansion.mjs` — stage cutaways, pixel inspection, and live matrix-product traces
- `src/motion.mjs` — interruptible spatial zoom, fallback, and reduced-motion handling
- `src/attention-flow.mjs` — connected single-head detail, causal comparison, and cancellable value-mixing animation
- `src/multi-head.mjs` — multi-head previews, feature splits, output concatenation, projection/residual, and merge animation
- `src/readout.mjs` — final encoder states, CLS trajectory, projector handoff, and symbolic decoder prefix
- `assets/` — original sample illustrations and icon
- `tests/engine.test.mjs` — numerical correctness tests
- `tests/bundle.test.mjs` — offline build syntax and dependency checks
- `tests/expansion.test.mjs` — cutaway, RGB indexing, and all-stage numerical trace tests
- `tests/motion.test.mjs` — geometry, transition lifecycles, interruptions, fallbacks, and reduced-motion tests
- `tests/attention-flow.test.mjs` — causal normalization, output equivalence, immutable encoder data, partial sums, and attention detail tests
- `tests/multi-head.test.mjs` — feature-split shapes, full-input projections, head order, exact merge/residual identities, and immutable comparisons
- `tests/readout.test.mjs` — seven-stage navigation, final-vector identities, CLS trajectory, patch ordering and delimiters, projector traces, and immutable symbolic generation
- `server.mjs` — dependency-free local HTTP server
- `build.mjs` — generates the self-contained `standalone.html`
- `standalone.html` — portable, fully offline website

After editing the source, regenerate the offline build with `node build.mjs`.

## Deploy

GitHub Pages is configured to deploy from **main / (root)**. Pushing updates to `main` triggers publication. The `.nojekyll` marker serves the static HTML, CSS, JavaScript modules, and SVG assets without Jekyll processing.

The hosted entry point is `index.html`; `standalone.html` remains available as the self-contained offline version. Run `node build.mjs` after source edits to refresh that offline artifact. Check the repository’s Actions tab for deployment status.

## Next step toward a full research explainer

Replace the untrained engine with a small pretrained ViT that exposes intermediate tensors, while keeping the same UI contracts. A browser inference backend would need the checkpoint plus attention/hidden-state outputs for every inspected layer. The present architecture intentionally separates the engine from the interface so that can be added without rebuilding the website.
