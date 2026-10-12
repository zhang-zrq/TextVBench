# TextVBench-Ref2V Evaluator & Metric Design and Implementation

## 1. Goal

TextVBench-Ref2V is a benchmark-specific evaluator whose behavior is aligned
with T2V where metric definitions are shared. Both evaluators live in the same
`textvbench` package while keeping their runtime modules and contracts separate.

The evaluator follows a shared three-layer architecture:

```text
Layer 1  Data / Ground Truth
        ↓
Layer 2  Understanding & Matching
        ↓
Layer 3  Metric Primitives
```

The main difference between T2V and Ref2V is the source of ground truth:

```text
T2V:
prompt / structured spec
    ↓
<target visual text, carrier>

Ref2V:
reference image(s)
    ↓
runtime OCR on each reference image
    ↓
<slot_id, target visual text, carrier, ref_id, reading_order>
```

OCR preprocessing, marker-image construction, candidate filtering, target
batching, VLM invocation, normalization, frame sampling, retry logic, and
metric aggregation follow the T2V conventions, but are implemented locally in
Ref2V. Ref2V's public evaluator input is the release dataset described in the
I/O specification; intermediate OCR manifests are not inputs.

---

## 2. Repository Integration

Current repository:

```text
TextVBench/
├── data/{T2V,Ref2V}/
└── src/textvbench/evaluators/{t2v,ref2v}/
```

Ref2V mirrors relevant T2V behavior through its own evaluator modules and does
not import the T2V implementation at runtime.

Conceptually:

```text
T2V-compatible local evaluation components
        ↓
Ref2V reference-OCR input adapter
        ↓
Optional Ref-ID Matching
        ↓
Local reference-grounded target-candidate matching
        ↓
Shared metrics
        ↓
Ref2V-specific Reading Order metric
```

---

# 3. Layer 1 — Data Input / Ground Truth

## 3.1 Unified Target Abstraction

The core evaluation unit remains a target visual text associated with a carrier.

Recommended logical representation inside the evaluator:

```json
{
  "slot_id": "T0001",
  "target_text": "ZERO SUGAR",
  "carrier": "complete visual carrier in this reference image",
  "ref_id": "R01",
  "carrier_id": "C_R01",
  "reading_order": 2
}
```

Required logical slot fields:

```text
slot_id
target_text
ref_id
carrier_id
reading_order
```

Derived / matching metadata:

```text
carrier
position (bbox center)
```

### `ref_id`

`ref_id` identifies the reference image / reference instance from which the
slot is defined. Slot IDs are assigned by the evaluator in a single sequence
across all references in a sample. The merged manifest does not contain a
precomputed target list. In the Ref2V input JSONL, `carrier` and `position` are
not required fields; the evaluator derives the carrier as the complete
reference image and the position from the OCR bbox center.

Use generic IDs:

```text
R01
R02
R03
...
```

Do not hard-code semantic assumptions such as:

```text
front
back
left
right
```

A `ref_id` may represent:

- another view of the same product;
- another product;
- another document;
- another screen / UI;
- another reference instance.

### Multiple slots sharing one carrier

This is expected and should not be treated as a special case.

Example:

```text
ref_id = R01
carrier = product package

T01 = Brand Name
T02 = ZERO SUGAR
T03 = 330 ML
T04 = ORIGINAL
```

The carrier is a semantic / spatial constraint, not a unique slot identifier.
For the current implementation, a reference image is treated as one complete
visual carrier (`C_<ref_id>`), so all visible OCR text in that image belongs to
that carrier. A reference OCR box produces exactly one slot.

---

## 3.2 Ref2V Benchmark Sample Structure

Conceptually:

```json
{
  "sample_id": "0001",
  "prompt": "...",
  "reference_images": [
    {
      "ref_id": "R01",
      "filename": "reference.png"
    }
  ]
}
```

`filename` is resolved under the benchmark file's sibling `imgs/` directory.
Reference OCR runs when evaluation starts; its texts, polygons, and reading
order are not pre-saved in the benchmark manifest. The target's position is the
reference OCR bbox center, while the generated position is derived from the
matched OCR bbox center. These positions are used internally for reading-order
calculation and are not included in the compact VLM target payload.

For VLM understanding, each reference image is rendered with its `ref_id`
(`R01`, `R02`, ...) in a white band at the upper-left corner and then resized
proportionally to a maximum longest edge of 256 pixels. The original image is
used for runtime OCR; the labeled compact image is used for Ref-ID routing,
target-candidate grounding, and the corresponding debug artifact.

---

# 4. Layer 2 — Understanding & Matching

Layer 2 contains two stages:

```text
Stage A: Ref-ID Matching      # only for multi-reference samples
Stage B: Target-Candidate Matching
```

---

## 4.1 Stage A — Ref-ID Matching

### Activation rule

```text
number of reference images == 1
    → SKIP Ref-ID Matching

number of reference images > 1
    → RUN Ref-ID Matching
```

This should be a hard evaluator rule.

### Single-reference case

If only one reference image exists:

```text
reference_images = [R01]
```

do not call an extra VLM.

Directly set:

```text
active_ref_ids = [R01]
```

and continue with target-candidate matching using all R01 reference-OCR
slots.

### Multi-reference case

When multiple reference images exist:

```text
R01, R02, ..., Rn
+
generated video frame
        ↓
Ref-ID Matching VLM
        ↓
matched_ref_ids
```

The output must support zero, one, or multiple IDs:

```json
{
  "matched_ref_ids": ["R01", "R03"]
}
```

This is necessary because one frame may contain multiple products, documents, screens, or other reference instances.

### Purpose

Ref-ID Matching is a routing stage:

> Determine which reference image(s) the generated frame corresponds to, then activate only the slots associated with those reference IDs.

It is not an OCR stage and does not directly compute text metrics.

### No-match case

The matcher may return:

```json
{
  "matched_ref_ids": []
}
```

Record this status explicitly.

Do not silently treat the frame as successful or discard the information. The
frame is marked `evaluated: false`, excluded from metric and temporal
aggregation, and its OCR candidates are not counted as evaluated extra text.
If every sampled frame has no active reference, the video's final metrics are
forced to zero.

For v1, Ref-ID matching itself is not a separate leaderboard metric, but the
intermediate result is preserved in debug frame observations for later
analysis or custom metrics.

---

## 4.2 Target Activation

After Ref-ID routing:

```text
matched_ref_ids
      ↓
activate slots whose ref_id is in matched_ref_ids
      ↓
construct unresolved slot batches
```

In target-candidate matching, `target.ref_id` is the carrier-routing key. The
VLM must map that key to the reference image carrying the same visible `Rxx`
label, find the corresponding carrier in the generated marker image, and only
then choose its `Pxxx` OCR candidate tags. `Rxx` reference IDs, `Txxxx` slot
IDs, and `Pxxx` generated OCR candidate IDs are separate namespaces.

Single-reference:

```text
R01
 ↓
all R01 slots
```

Multi-reference:

```text
[R01, R03]
 ↓
slots(R01) ∪ slots(R03)
```

Use the same target batching strategy as T2V.

---

## 4.3 Stage B — Target ↔ OCR Candidate Understanding Matching

Mirror the established T2V behavior with Ref2V-local code:

```text
Generated frame
    ↓
OCR
    ↓
candidate filtering
    ↓
marker image
    ↓
small batch of unresolved targets
    ↓
VLM candidate selection
    ↓
slot → selected OCR candidate IDs
```

Core principle:

> This remains candidate selection, not free OCR.

Maintain behavior parity with T2V for:

- OCR candidate generation;
- OCR normalization;
- marker-image visualization;
- target batching;
- VLM matching;
- unmatched representation;
- selected candidate ID ordering;
- repetition / hallucination candidate accounting;
- retry and parsing logic.

The Ref2V matching prompt should be a generalized version of the existing T2V prompt rather than a new matching philosophy.

Each matching request contains at most
`matching.target_batch_size` active slots. The marker image is regenerated for
each batch and shows only candidates that remain available after previous slot
assignments. This keeps the visual marker consistent with the structured
candidate list and prevents candidate reuse across batches. The default batch
size is 3; a value of 1 is a valid slower, more conservative configuration.
Within one request, the same OCR candidate may be selected for multiple slots;
this accommodates a generated-frame OCR box that merges adjacent visual texts.
Such a candidate is removed before the next request batch.

The structured target payload is intentionally compact and contains exactly:

```json
{
  "slot_id": "T0004",
  "target_text": "Xiaoran Zhou 4:05 PM",
  "ref_id": "R01"
}
```

The structured candidate payload contains exactly `id`, `ocr_text`, and
`confidence`. Candidate geometry is available only in the marker image. The
matching response returns `{ "slot_id": "...", "selected_ids": [...] }`
for each target, so response validation does not depend on a transient batch
index.

---

# 5. Layer 3 — Metric Primitives

The implementation exposes a fixed, compact Ref2V v1 metric set in each
sample's `result.json` and in successful entries of `summary.json`. The pure
metric functions remain reusable for future aggregate or custom scoring, but
the evaluator's default output uses the definitions below.

---

## 5.1 Per-target Spatial Fidelity

For every reference OCR slot, compute the normalized edit similarity between
its target text and the concatenated text of the generated candidates selected
for that slot in each evaluated frame. Average these frame scores per slot.
Missing selected candidates contribute zero.

The public result exposes this as:

```text
metrics.per_target_spatial_fidelity[slot_id]
```

Reference OCR text is not loaded from a separate target manifest; it is
discovered at evaluation runtime.

---

## 5.2 Per-target Temporal Fidelity

Temporal fidelity compares the selected text for the same slot in adjacent
evaluated frames. A missing selection in either side of a pair contributes
zero. Frames for which Ref-ID routing returns no active reference are excluded
from both metric and temporal aggregation.

The public result exposes this as:

```text
metrics.per_target_temporal_fidelity[slot_id]
```

---

## 5.3 Length-weighted Spatial and Temporal Fidelity

Within each reference, slot scores are weighted by the normalized character
length of the reference target text. Reference-level scores are then averaged
with equal weight across references, so an inactive or entirely unmatched
reference receives zero credit rather than disappearing from the denominator.

The public result exposes:

```text
metrics.length_weighted_spatial_fidelity
metrics.length_weighted_temporal_fidelity
```

If no frame matches any reference in the video, all fidelity metrics are zero.

---

## 5.4 TextVScore

`TextVScore` is the harmonic mean of length-weighted spatial and temporal
fidelity, reported on a 0--100 scale:

```text
TextVScore = 100 * H(spatial_fidelity, temporal_fidelity)
```

It is emitted as `metrics.textv_score`.

---

## 5.5 ExtraTextRatio

After candidate filtering and slot matching, active OCR candidates that remain
unassigned are treated as extra text. Duplicate and candidate-cap exclusions
are included in the extra-text accounting. No-match frames are excluded from
the average.

The result is emitted as `metrics.extra_text_ratio`.

---

## 5.6 Reading Order Fidelity — Ref2V metric

Reference reading order is assigned by sorting the centers of reference OCR
boxes by `(center_y, center_x)`, producing a sequence of slot IDs. Generated
reading order is computed after target-candidate matching by sorting the centers
of matched generated OCR boxes with the same key.

The comparison uses pairwise order accuracy. Missing slots are appended to the
end of the predicted sequence rather than removed, so a missing slot remains a
reading-order error. Only slots belonging to the same reference are compared.

Reading order is computed after understanding matching and never influences
the matching decision. The result is emitted as
`metrics.reading_order_score` and per-reference `refs[].reading_order`.

---

# 6. Metrics Excluded from Ref2V v1

Do not implement the following as core Ref2V metrics in v1:

```text
Reference Layout Fidelity
Reference Appearance / Structure Preservation
pixel-level reference similarity
bbox IoU between reference and generated frames
3D / geometric surface consistency
```

They are either difficult to define robustly across video motion or outside the central goal of TextVBench.

---

# 7. Multi-Reference Support

The code structure should support multi-reference samples even if the first released data is mostly single-reference.

Use generic IDs:

```text
R01
R02
R03
...
```

The evaluator should only care about:

```text
generated frame
    ↓
which ref_id(s) are active?
    ↓
which targets should be evaluated?
```

The semantic relationship between references belongs to dataset metadata, not evaluator core logic.

---

# 8. End-to-End Flow

```text
Ref2V Sample
│
├── prompt
├── reference image(s)
└── reference OCR slots grouped by ref_id
        ↓
Generated Video
        ↓
Frame Sampling
        ↓
┌─────────────────────────────────┐
│ Reference count == 1 ?          │
├─────────────────────────────────┤
│ YES → directly activate R01     │
│ NO  → Ref-ID Matching VLM       │
└─────────────────────────────────┘
        ↓
active / matched ref_id(s)
        ↓
activate corresponding reference OCR slots
        ↓
OCR + candidate filtering
        ↓
marker image
        ↓
Target ↔ Candidate VLM Matching
        ↓
frame-level matching result
        ↓
Metric Primitives
├── Per-target Spatial Fidelity
├── Per-target Temporal Fidelity
├── Length-weighted Spatial Fidelity
├── Length-weighted Temporal Fidelity
├── TextVScore
├── ExtraTextRatio
└── Reading Order Score
        ↓
video-level result and batch summary
```

---

# 9. Behavior Alignment with TextVBench-T2V

Ref2V follows the relevant `TextVBench-T2V` conventions wherever possible,
but owns independent implementations of every runtime component. The two
evaluators share only the top-level package and CLI dispatch.

Priority behavior-alignment targets:

```text
1. OCR inference and normalization
2. OCR candidate schema
3. marker-image generation
4. VLM target batching
5. VLM request / retry logic
6. target-candidate output parsing
7. frame sampling
8. per-frame → per-video aggregation
9. existing metric definitions
10. result schema and CLI conventions
```

The current independent implementation contains these local components:

```text
A. Ref2V merged-input and runtime reference-OCR adapter
B. optional Ref-ID matcher
C. ref_id-based slot routing
D. reference-grounded target-candidate matching
E. Reading Order metric
F. compact summary and run-details output
G. read-only evaluator Review API and Web UI
```

---

# 10. Implementation Status / Historical Order

The following sequence records the intended implementation order. All five
phases are implemented in the current v1 evaluator; it is retained as design
history rather than as a pending work plan.

```text
Phase 1
Single-reference Ref2V
→ mirror T2V target-candidate matching behavior locally
→ verify shared metrics

Phase 2
Add reading_order
→ implement Reading Order Fidelity

Phase 3
Enable multi-reference
→ add Ref-ID Matching
→ route targets by matched_ref_ids

Phase 4
Regression testing
→ confirm T2V and Ref2V preserve consistent metric behavior

Phase 5
Evaluator review and observability
→ expose reference, OCR, routing, matching, reading-order, and temporal evidence
→ provide a read-only bilingual Web UI
→ sanitize paths, credentials, and VLM endpoints at the API boundary
```

---

# 11. Evaluator Review and Evidence Contract

The implemented Review service consumes the evaluator's existing public and
debug artifacts. It does not rerun OCR/VLM calls, modify matching decisions,
or recompute the official result. Public `<sample_id>/result.json` metrics take
precedence over the debug copy, while debug `result.json` and frame artifacts
provide the evidence used to explain those metrics.

For each sample, the interface exposes:

- global, per-reference, and per-target scores;
- labeled compact reference images and runtime reference-OCR slots;
- the generated video when explicitly configured, plus all sampled frames;
- raw, active, filtered, duplicate-extra, and capped-extra OCR states;
- multi-reference Ref-ID routing traces, or an explicit skipped state for a
  single-reference sample;
- target-candidate matching batches and candidate reuse across targets in the
  same batch;
- sanitized VLM system prompts, payloads, attempts, responses, and final
  errors;
- per-frame reading-order sequences, missing slots appended at the end, and
  incorrect slot pairs;
- per-target temporal observations across evaluated frames;
- sanitized result, manifest, input, configuration, and run details.

A failed attempt followed by a successful retry is reported as a retry, not as
a failed case. A trace is considered failed only when its final outcome fails.
Frames with no active reference are visibly marked as excluded, matching the
metric aggregation rule.

The Review API is read-only and only serves allowlisted artifacts beneath the
configured roots. It removes credentials, VLM endpoints, and machine-specific
absolute paths before returning JSON. Video files must be explicitly supplied
through `--video-dir` or `--video-manifest` and are served with HTTP Range
support.

---

# 12. Core Design Principle

```text
Data defines WHAT should exist.

Understanding Matching determines WHICH generated OCR region corresponds to each target.

Metrics determine HOW the matching result is scored.
```

Keep these responsibilities separate.

The evaluator should preserve intermediate matching results so users can recompute or reweight metrics without rerunning OCR / VLM understanding whenever possible.
