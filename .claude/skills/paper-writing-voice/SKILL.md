---
name: paper-writing-voice
description: The author's (Hunter Schweiger's) scientific-manuscript prose style — how to draft and edit paper PROSE in his voice, for the .tex manuscript in the NeuralPredictor repo and any results/intro/discussion/abstract text. Distilled from his co-authored Stem Cell Reports 2026 organoid paper (the reference corpus) and refined via a living preferences log at the bottom that grows as we draft the new Nature Neuroscience predictor paper. Covers section-level templates (gap-first abstract, funnel intro, finding-as-header results with a motivation→method→result→interpretation microstructure, thesis-first discussion with an explicit limitations paragraph), sentence-level voice (first-person-plural active, "To X, we did Y" openers, calibrated hedges, inline stat/figure-callout formatting), and which patterns are transferable voice vs venue/content-specific. Invoke whenever writing or editing any paper TEXT — "write up", "write the results", "draft the intro/discussion/abstract", "help with the writing", "work on the paper / manuscript text", "reword this", "rephrase", "tighten this", "clean up the wording", "word this", "make it sound like me", "in my voice / my style" — or when editing a sections/*.tex, supplementary .tex, or a figure caption. The trigger is the writing task itself, not the word "prose": match on write / writing / wrote / draft / reword / rephrase / revise / edit-the-text / word-this / manuscript / paper-text / results-section / intro / introduction / discussion / abstract / caption / .tex. NOT the numbers-ledger doc in the analysis repo (that is `results-numbers-doc`, the upstream source of the numbers this voice renders) and NOT figures (those are `figure-style` / `neural-predictor-figures`).
argument-hint: [section or paragraph to draft/revise, or a paste of draft prose to bring into voice]
---

# paper-writing-voice — write manuscript prose in Hunter's style

This skill owns **how the sentences read**, not which numbers are true. The division of labor:

- **`results-numbers-doc`** (analysis repo) = the ledger: which number is real, its source
  CSV+column, its caveats. It is the trustworthy *input*. Pull numbers FROM it; never invent a
  number in prose, and never let a prose paragraph become the place a number first appears.
- **this skill** = how a chosen number gets phrased into the `.tex`. Producer → consumer.
- **`figure-style` / `neural-predictor-figures`** = the figures. ⛔ No prose belongs baked into a
  figure image (see project memory `feedback_no_prose_in_figures`); a LaTeX `\caption` is external
  editable prose and IS in this skill's scope, a title/subtitle inside a `.png` is not.

**The manuscript lives in a separate repo**: `/Users/hunterschweiger/Desktop/braingeneers/NeuralPredictor/`
(sections in `sections/*.tex`, supplement in `supplementary/`). Edits there are folded into the
user's own commits — don't commit the manuscript repo unless asked.

🔀 **Concurrent editing is normal here.** Hunter runs several sessions editing DIFFERENT sections
of the same `results.tex` at once (e.g. an electrode-decoding session and a cartpole session). So:
confine your edits to your assigned section, keep every Edit `old_string` unique to that section,
re-read immediately before each write (the harness forces a re-read if the file changed under you,
which prevents silent clobbering), and never reflow or renumber across a section boundary. Two
sessions in different sections cannot corrupt each other this way; the only unsafe move is editing
the same lines another session is in.

## Who owns the final voice — and how this skill is used

The reference paper's own AI-disclosure states the authors used AI "to improve language clarity
and readability" and "reviewed and edited" everything, taking "full responsibility for the final
version." That is exactly the intended workflow here: **I draft in this voice; the user owns and
reworks the final prose.** So:

- When the user asks for a full-scale rewrite ("just do your best, this should be a good draft"),
  draft confidently in the voice below.
- When updating an existing manuscript where the user is doing the prose pass themselves (the
  default per the fig-update runbook), **change as little as possible and flag rather than guess** —
  the runbook's §3 governs, this skill only says what "in his voice" means for the words you do
  touch.
- Leave a short note flagging anything you're unsure of rather than silently committing a guess.

## The reference corpus

Two genuine Hunter-voice samples ground this skill:

1. **The Stem Cell Reports paper** — Hernandez, Schweiger, et al., "Establishing mouse forebrain
   organoids as models of intrinsic cortical network assembly," *Stem Cell Reports* 21, 102832
   (2026). Local PDF (stable, in-repo): `/Users/hunterschweiger/Desktop/braingeneers/NeuralPredictor/my_old_papers/ms_org/Establishing mouse forebrain organoids as models of intrinsic cortical network assembly.pdf` (DOI `10.1016/j.stemcr.2026.102832`). Hunter is co-first author and wrote
   essentially all of it; AI help was light and from earlier, weaker models. **Treat it as genuine
   personal voice, not house style.**
2. **A fellowship proposal** on the current paper's exact topic (foundation model of neural
   dynamics, selection vs instruction, closed-loop organoid learning), saved at
   `reference/fellowship_sample.md` in this skill. Unpublished — do not distribute or quote outside
   this work. It is the closest sample to the paper we are drafting, so **weight it most heavily
   for voice**; its signature moves are distilled in the 2026-08-24 log entries below.

⚠️ The remaining caveat is content/venue, not authorship: the paper is **wet-lab organoid biology
in Stem Cell Reports** and the fellowship is a **future-tense proposal**, while the new paper is
**computational/ML neuroscience results in Nature Neuroscience**. Separate the transferable
rhetorical skeleton (below) from the content/venue-specific surface (IHC-marker reporting, `± SD`
cell counts, Summary-not-Abstract, proposal "I will…" framing).

---

# The voice, distilled (with verbatim exemplars)

> **North star (Hunter's framing):** it is an art to make a difficult concept technically rigorous
> and publication-grade *while* keeping it understandable to a general neuroscientist/biologist —
> Nature Neuroscience's actual reader, not an ML specialist. Every technical paragraph is judged on
> this. The concrete technique is in the 2026-08-24 "guiding principle" log entry; the short of it:
> keep the precise term but ride it in apposition behind a plain-language spine, convert operations
> into their meaning, and never drop rigor (n, P, effect, formal term all stay) to buy readability.

## Macro-structure

### Abstract / Summary — gap-first, five moves in one dense paragraph
The reference Summary is ~180 words, one paragraph, and runs: **(1) field + gap → (2) the system →
(3) what we did → (4) findings → (5) significance framing.** Verbatim:

> "The mouse cortex is a canonical model for studying how functional neural networks emerge, **yet
> it remains unclear which** topological features arise from intrinsic cellular organization versus
> sensory input. [system:] Mouse forebrain organoids provide a powerful system to investigate these
> intrinsic mechanisms. [what:] **We generated** dorsal (DF) and ventral (VF) forebrain organoids
> … [findings…] [significance:] **Our findings demonstrate how** cellular composition influences
> neural circuit self-organization, **establishing** mouse forebrain organoids as a tractable
> platform to study cortical network architecture."

The signature move is opening on an established truth, then pivoting on **"yet it remains unclear /
remains poorly understood"** to the gap. Close on **"Our findings demonstrate … establishing X as
a platform to …"** (significance stated as what the work *enables*, not as a boast).

⚠️ Venue: Stem Cell Reports calls it **Summary**; Nature Neuroscience calls it **Abstract** and
caps it ~150 words with no subheads — same five moves, tighter.

### Introduction — a funnel that ends on "Here, we…"
Five paragraphs, broad → specific: (1) the big concept stated plainly ("The brain is organized as
a complex network whose topology supports efficient computation"); (2) what's known (dissociated
cultures + *in vivo*); (3) prior efforts on this system **and their limitations**; (4) the gap
stated flatly — **"protocols that yield … are still lacking," "the developmental processes that
generate this topology remain poorly understood"**; (5) the pivot:

> "**Here, we address a key gap in the field by** systematically investigating how regional cellular
> composition drives the formation of neural networks independent of sensory input."

Rules: every paragraph earns the next; the gap is named in plain words before "Here, we"; "Here,
we" states scope and approach in one sentence, not a findings list.

### Results — the finding IS the header, and each block has a fixed microstructure
**Subsection headers are full declarative sentences stating the result**, not topic labels:

> ✅ "VF organoids develop stronger small-world topology through enhanced local clustering"
> ✅ "DF and VF organoids exhibit distinct network dynamics"
> ⛔ not "Small-world analysis" / "Network topology"

Each results block follows **motivation → method-as-question → result+stats → interpretation**:

1. **Motivation**, usually anchored to a known *in vivo* fact:
   "*In vivo*, mouse cortical networks acquire small-world topology during early postnatal
   development before sensory experience (refs)."
2. **Method framed as the question it answers** — the signature **"To [verb] whether … , we …"**:
   "**To determine whether** organoids recapitulate this trajectory …, **we analyzed** small-world
   organization across DF and VF development."
   "**To investigate whether** E-I balance influences network dynamics, **we** pharmacologically
   manipulated their synaptic transmission."
3. **Result with numbers inline** (see stat formatting below).
4. **One calibrated interpretive sentence**, often re-tying to *in vivo*:
   "suggesting insufficient inhibitory tone may cause decorrelation."

### Discussion — thesis first, contextualize, then an explicit limitations paragraph
Opens with a one-sentence thesis: **"Our study shows that mouse forebrain organoids can
self-organize into complex neural networks that recapitulate several organizational principles of
cortical development."** Then each finding is placed against literature/*in vivo*. Contains a
dedicated, signposted limitations paragraph — **"Several limitations should be considered when
interpreting our findings."** — that states what the data cannot claim, then closes forward-looking
("establish … as a powerful platform for dissecting how …").

## Sentence-level voice

- **First-person-plural, active, past tense for what was done**: "We generated…", "We tested…",
  "We analyzed…". Not "It was found that."
- **"To X, we Y" is the workhorse opener** for any analysis. Reach for it before "We did Y to see
  X." The question leads.
- **Calibrated hedges, chosen deliberately** — the interpretive verb carries the confidence level:
  `suggesting` / `consistent with` / `pointing to` / `may reflect` / `indicating`. A correlational
  result never gets a causal verb. The paper says it outright:
  "while the observed differences … strongly correlate with inhibitory neuron enrichment,
  establishing causality … will require future targeted manipulations." **State what the design
  cannot support, don't imply past it.** (Reinforced by memory `feedback_stats_discussion_style`,
  `feedback_targeted_drop_is_a_diagnostic_not_a_test`.)
- **No over-dramatization, no sycophancy in the prose** — findings are stated, not sold. Avoid
  "strikingly," "remarkably," "surprisingly" unless the result genuinely is. (Memory
  `feedback_stats_discussion_style`.)
- **Contrastive two-arm framing as a through-line.** The whole organoid paper is DF-vs-VF; nearly
  every block reports the two arms side by side. The new paper's analogues (model-vs-baseline,
  evoked-vs-sustained, causal-vs-spontaneous, dorsal-vs-ventral) should be carried the same way —
  name both arms in the same sentence with matched phrasing.
- **"Bracket / complementary" framing device** for two systems spanning a spectrum: "DF and VF
  organoids bracket the physiological state and provide complementary platforms." Useful whenever
  two conditions flank a reference.
- **Abbreviation discipline**: define once at first use (`dorsal (DF)`, `interneurons (IN)`,
  `excitatory-inhibitory (E-I)`), then use the abbreviation exclusively. Don't redefine.

## Numbers, stats, and callouts in prose

- **Inline parenthetical stats, both arms, then the p**: `(DF = 57.78 ± 38.20%; VF = 16.64 ±
  17.81%; p < 0.001)`. Value ± spread per arm, semicolon-separated, p last.
- **Name the correction, don't just star it**: "* significant after Bonferroni correction, p <
  0.017"; "mixed-effects model." The reader learns the test from the prose/legend, not a bare
  asterisk.
- **Figure callouts are parenthetical at the end of the clause they support**: "(Figure 1C, S2A,
  and S2B)", "(p < 0.001; Figure 5B)". Never mid-clause; never a sentence whose only content is a
  pointer.
- **Report the effect, not a derived ratio, as the headline** — lead with the primary quantity
  (×chance, slope, AUC), not a retention/percentage-of derived from it. (Project rule, memory
  `feedback_lead_with_effect_not_derived_ratio`.)
- **A paired change is its own number**, not the difference of two reported medians — if you state
  a paired delta, label it as paired. (Memory `paired_delta_vs_difference_of_medians`.)
- **Name the metric family precisely** — this project has several bits/AUC quantities with opposite
  rankings; never write "bits/spike" or "AUC" unqualified. (Memory `feedback_name_the_metric_family`,
  `shank3wt_discrim_survives_ft`.)
- Typographic conventions from the reference: *in vivo* / *ex vivo* italic; gene names italic
  (*Gad2*, *Pvalb*, *Camk2a*); protein/marker roman with superscript (`Pvalb+`, `Sox2+`); `n`
  italic; `p` italic.

## Transferable vs. don't-blindly-port

| Transferable (the rhetorical skeleton) | Content/venue-specific — re-fit, don't copy |
|---|---|
| gap-first abstract; "yet it remains unclear" pivot | "Summary" heading & length (NN = "Abstract", ~150 w) |
| funnel intro ending on "Here, we…" | *in vivo* mouse-cortex anchoring of every block |
| finding-as-header results | IHC-marker / `± SD` cell-count reporting style |
| motivation→method→result→interpretation blocks | organoid-specific jargon (DF/VF, Nkx2.1, WFA) |
| "To X, we Y" openers; calibrated hedges | Stem Cell Reports reference-density (many cites/claim) |
| explicit limitations paragraph; thesis-first discussion | wet-lab methods voice |
| contrastive two-arm / bracket framing | |

---

# Preferences log (living — append as we draft the new paper)

**How to use this section:** every time the user corrects a draft, states a preference, or
confirms a phrasing, add a dated one-line entry here rather than rewriting the distilled-voice
sections above. The sections above are the *derived* style from the published paper; this log is
the *authoritative* running record of Hunter's personal preferences for the new paper, and **it
wins over the sections above wherever they conflict.** Keep entries terse: the rule + a one-clause
why. Promote a repeated preference up into the distilled sections only once it's stable.

Entry format: `- [YYYY-MM-DD] <rule>. — <why / source>`

- [2026-08-24] Seed from project memory (writing-relevant, pre-confirmed): lead with the effect
  (×chance/slope), never a derived retention %; a paired change is its own labeled number, not a
  difference of medians; name the metric family (bits/AUC are ambiguous in this project); calibrated
  hedges, no over-dramatization, no sycophancy; state what the design can't claim rather than
  implying causation. — memory `feedback_lead_with_effect_not_derived_ratio`,
  `paired_delta_vs_difference_of_medians`, `feedback_name_the_metric_family`,
  `feedback_stats_discussion_style`.
- [2026-08-24] Skill created; reference corpus = the Stem Cell Reports organoid paper. Refine
  toward Hunter's personal voice here as the Nature Neuroscience predictor paper is drafted. — this
  session.
- [2026-08-24] ⛔ Never end a results subsection (or any section) on a bare negative / scope
  caveat — "we note we did not test X," "this is only a geometric statement," etc. A terminal
  caveat leaves the reader on a negative feeling. Resolve the block on the finding + its forward
  interpretation; demote a needed scope caveat to a mid-paragraph subordinate clause ("a
  geometric, not longitudinal, claim") or move it to the Discussion's limitations paragraph. The
  last sentence of a block is prime real estate — spend it on the payoff, not the disclaimer. —
  Hunter, this session (flagged the cartpole subsection's "we did not test whether the readout
  existed before first exposure" ending).
- [2026-08-24] ⭐ Frame a result by the CANONICAL question it answers + the system property that
  makes it uniquely answerable — not as an isolated organoid/ML finding. Two moves, both from the
  reference paper's engine: (1) name the mainstream-neuro debate and its landmarks (e.g.
  selection vs instruction → Edelman selectionism, Buzsáki's preconfigured brain, the in-vivo
  spontaneous→evoked correspondence of Luczak 2009 / Berkes 2011) so the result reads as a
  controlled test of a known hypothesis; (2) pivot on the "not feasible in vivo" differentiator —
  here, that no animal is ever a blank slate and you cannot inject defined information into a naive
  brain and read its full response, so the organoid + latent ruler supply the missing controlled
  test. Put the "why it matters / why only we can do this" as the PIVOT of the framing paragraph,
  not a buried clause; demote pure measurement-obstacle framing ("lacks a common reference frame")
  to the secondary obstacle the method solves. ⚠️ Keep it a convergent/controlled test, not "we
  alone solved it" — don't overclaim a geometric result into a settled longitudinal one. — Hunter,
  this session (cartpole selection-vs-instruction subsection felt "disconnected from mainstream
  neuro").
- [2026-08-24] ⛔ NEVER use em dashes (— or LaTeX `---`). They read as AI-generated. Recast with
  commas, a colon, parentheses, or two sentences. Applies to prose AND figure captions. (En dashes
  for numeric ranges, `$0.5$--$8$~Hz, are fine.) — Hunter, this session.
- [2026-08-24] ⛔ Avoid the AI "it's not X, it's Y" / "not just X, but Y" / "X isn't about Y, it's
  about Z" rhetorical template. It's a tell. State the positive claim directly. NOTE this does NOT
  ban a substantive scientific contrast that IS the finding (e.g. "recruits spontaneous dynamics
  rather than constructing new ones" = the selection-vs-instruction result) — the ban is on the
  empty rhetorical flourish, not on a real dichotomy the data adjudicate. — Hunter, this session.
- [2026-08-24] ⛔ When framing an OPEN debate, don't state one side's supporting evidence as
  settled fact and then call the question unresolved — it contradicts itself. Either present that
  evidence AS evidence ("consistent with X but unable to exclude Y, because…") or cut it. Preferred
  motivation shape for a "why only we can answer this now" frame: state the debate → name the
  concrete problem(s) that kept it unanswerable → map our solution(s) onto them one-to-one ("X and
  Y supply both"). The named-barrier must precede any "removes this barrier" sentence. — Hunter,
  this session (caught a "nobody knows / but in vivo we know" contradiction in the cartpole frame).
- [2026-08-24] ⭐ Keep it to the point — no background tangents unless strictly needed to set up
  the results. A motivating frame states the question, the barrier, and the solution, then stops;
  it does not survey the literature. Let a citation carry "the arguments exist" rather than
  narrating both sides. Prefer the shortest framing that makes the result land. — Hunter, this
  session (cut the in-vivo evidence sentence from the cartpole frame; citation covers it).
- [2026-08-24] ⛔ Frame the gap as an UNREAD QUESTION or a MISSING TOOL, never by disparaging the
  alternative system. "Two things the intact brain cannot offer," "no experiment can deliver…" reads
  as crapping on in-vivo work and is combative. Hunter's own register: "where biological learning
  falls … has not been read directly, because … has lacked a common description." Say what is
  missing (a transferable ruler, a direct readout); let the enabling property of the new system
  stand matter-of-factly ("grown without sensory input, organoids provide a definable spontaneous
  baseline"), not as the incumbent's failure. — Hunter, this session (softened the cartpole frame;
  source = his fellowship proposal on the same idea).
  ↳ REFINED same day: the in-vivo limitation IS fair to state when framed as a concrete,
  mechanistic CONFOUND rather than a blanket "can't" — the difference between an insult and a
  reason. Hunter's accepted phrasing: "by the time mouse electrophysiology can capture activity with
  enough resolution to probe these theories, the brain has already undergone some sensory-driven
  development, confounding efforts to isolate the contribution of spontaneous activity." Name the
  specific confound (timing/resolution vs sensory-driven development), not a value judgment. Don't
  over-correct the rule above into never mentioning in-vivo limits at all.
- [2026-08-24] ⭐ Voice patterns lifted from Hunter's fellowship proposal — a PURER personal-voice
  sample than the co-authored Stem Cell Reports paper (which is house/AI-assisted style). When in
  doubt about voice, weight this sample over the distilled sections above: (1) turn a debate into a
  measurable quantity — "This turns a long-running debate … into a quantity I can read off as
  geometry"; (2) map evidence to interpretation with a parallel "To the degree X …; to the degree
  Y …" construction (NOT "not X but Y"); (3) plain active verbs — read, map, project, reach,
  recruit; (4) selection stated as "recruited and amplified rather than created." The proposal is
  unpublished — ask before quoting it; a saved copy can live under this skill's `reference/` if
  Hunter approves. — this session.
- [2026-08-24] ⭐⭐ GUIDING PRINCIPLE — technical + legible is the whole game (see North star at
  top). Concrete technique for a jargon-heavy results paragraph: (1) keep the precise term but ride
  it in APPOSITION behind a plain-language spine — "an optimal-transport (Gaussian-Wasserstein)
  distance," "the main axes of its spontaneous activity (its leading principal components)"; the
  sentence must still read if the parenthetical is deleted. (2) Convert the operation into its
  MEANING, not its name — not "we computed the Wasserstein distance and decomposed it" but "we
  asked how far apart the two clouds of activity sat, and split that into a shift in the average
  state versus a change in its shape." (3) One concrete anchor per abstract quantity: a "cloud of
  states," an "axis," a "set-point." (4) Normalize numbers to intuitive units ("a value of one
  means no more than baseline drift"). (5) Rigor never leaves for readability — every n, P, effect,
  and the formal term stay, so a specialist still trusts it and can map to Methods. Dumbing-down is
  a failure; readability is bought in the setup and the verbs, not by removing the quantity. —
  Hunter, cartpole section ("an art to make difficult concepts technically [rigorous]... while
  making them understandable to a more general audience").
- [2026-08-25] ⭐ Prose must introduce and reference figure panels in the SAME ORDER they appear
  in the figure. A results subsection that discusses panel (f) before it has covered (b)--(e) reads
  as out of order and forces the reader to jump around the plate. If the narrative genuinely wants a
  different order, the fix is to REARRANGE THE FIGURE to match the prose, not to reference panels out
  of sequence, and that is a decision to raise with Hunter (it means re-laying-out the actual figure
  panels), never a silent call. Concretely for Fig.~2: the plate is b--e (in-distribution benchmark +
  the binning/data/width/ablation axes) THEN f--i (zero-shot + two-minute fine-tuning recovery), so
  the text runs benchmark+ablations first and the OOD fine-tuning payoff last. — Hunter, this session
  (the Fig 2 subsection had gone straight into fine-tuning (f--i) before the ablations (b--e)).
- [2026-08-25] ⛔ A figure/table callout is a bare POINTER, not a mini-caption. Strip any
  description of what the panel shows: `(...; representative cluster geometry in
  Fig.~\ref{fig:supp_y})` → `(...; Fig.~\ref{fig:supp_y})`. The sentence already conveys what the
  figure shows; if the panel just illustrates it, let the reader look. Keep a short lead-in ONLY
  when the callout points to a genuinely different analysis the sentence doesn't name (a control,
  an alternative representation), where the phrase is load-bearing. — Hunter, this session.
- [2026-08-25] ⛔ Don't call the stimulus "the drive" (noun). It reads as insider jargon a general
  neuroscientist won't parse. Say "the stimulation" / "the stimulation frequency" / "the stimulus."
  (The verb "stimulation drives the network" is acceptable, but prefer plainer verbs when handy.) —
  Hunter, this session.
- [2026-08-27] ⛔ Every main figure, supplemental figure, and table MUST be cited in the prose,
  with no exceptions. Within a section, panels must be referenced in the order they appear in
  the figure (a before b before c, etc.). If the narrative genuinely wants a different order, the
  fix is to rearrange the figure panels to match the prose (raise with Hunter), never to reference
  panels out of sequence. This also applies across figures: if Fig 5 is cited before Fig 4 has
  been introduced, the figure numbering or section order needs to change. — Hunter, this session.
- [2026-08-27] ⛔ Don't say "spontaneous repertoire." Too abstract. Say "spontaneous activity"
  (or "spontaneous dynamics" where the context is about time-varying structure). — Hunter, this
  session (Opus 5 draft introduced "repertoire" everywhere).
- [2026-08-27] ⛔ Banned words/phrases — these are either too abstract, too jargony, or don't
  map to how a general neuroscientist thinks about the concept. Use the replacements:
  - **"containment"** → "falls inside" (for the geometric fact), "reuses" (for the concept).
    Noun form if needed: "subspace overlap." "Containment" sounds like trapping, not shared
    structure.
  - **"retention"** (subspace context) → "variance captured" or "variance shared." "Retention"
    sounds like memory/storage, not geometric projection.
  - **"repertoire"** → "spontaneous activity" (see entry above).
  - **"drive" / "driven" / "drive rate" / "drive frequency"** → "stimulation" /
    "stimulation-evoked" / "stimulation frequency." Insider jargon; a general neuroscientist
    won't parse "the drive" or "driven activity." The verb form ("stimulation drives the
    network") is acceptable. Say "stimulation-evoked activity" or "stimulation activity," not
    "driven activity." (Reinforces the 2026-08-25 entry banning "the drive" as a noun.)
  - **"sustained readout"** → "uncorrected readout." In electrophysiology "sustained" means
    a response lasting throughout a stimulus; here it meant "reading absolute position without
    baseline subtraction," which is a different concept entirely. The figure labels are
    "Drift removed" / "With drift"; the prose pair is "baseline-subtracted readout" /
    "uncorrected readout." Note: "evoked readout" is NOT banned because "evoked" is used
    legitimately throughout (evoked response, evoked activity, etc.); prefer
    "baseline-subtracted readout" when labeling the contrast pair but don't mechanically
    replace every "evoked."
  - **"resting block" / "resting activity" / "resting geometry"** → "spontaneous block" /
    "spontaneous activity" / "spontaneous geometry." Organoids aren't "at rest" the way a
    subject in an fMRI scanner is; calling unstimulated epochs "rest" borrows a human-imaging
    convention that misleads here. "Spontaneous" is the standard electrophysiology term for
    activity in the absence of stimulation. Internal code variable names (e.g. `rest_block`,
    `rsel`) are exempt.
  - **"session"** (when referring to a multi-day continuous recording) → **"experiment."**
    "Session" is ambiguous (coding session? recording session?) and inconsistent with the rest
    of the paper, which calls these experiments. Use "experiment" throughout.
  - **"epoch"** (when referring to a recording interval) → **"recording"** (e.g. "spontaneous
    recording," "stimulation recording"). "Epoch" is ML jargon (training epoch) and will
    confuse biologist readers. The one exception: literal ML training epochs in methods.
  - **"block"** (when referring to a spontaneous or stimulation recording interval) →
    **"recording."** The paper's established terminology is: a **cycle** is one pass through
    all five stimulation frequencies with interleaved spontaneous recordings; each interval
    within a cycle is a **recording** (spontaneous recording, stimulation recording), not a
    block. Internal code variable names (e.g. `stim_block`) are exempt.
  - **"campaign" / "measurement campaign"** → **"comparison"**, or state the fact directly.
    Each one is a single ASD line against its own control, prepared and recorded together:
    \textit{SHANK3}$^{+/-}$ versus \textit{SHANK3}$^{+/+}$, and SF1755 versus CW20107.
    "Comparison" already names that, so "campaign" is a second word for one object — which
    is why it reads as a separate concept the reader then goes looking for and cannot find.
    The giveaway is that the paper's own sentence used both at once: "the two COMPARISONS
    were prepared, recorded, and spike-sorted in entirely separate measurement CAMPAIGNS."
    ⚠️ Where the point is specifically that the two were measured apart, SAY THAT rather
    than naming a category for it — "share no plate, rig, or recording date" is the actual
    claim and is what rules out a batch effect; "separate campaigns" only gestures at it.
    ⛔ Do not swap in "cohort" (the paper already uses it for the drug-clean holdout),
    "batch", or "preparation" (already means 3D organoid / 2D dissociated / slice). Internal
    code exempt — `ARMS`, "the SF1755 arm" etc. stay as they are.
  - **"tissue"** (as the ACTING SUBJECT of a result) → **"network."** "The tissue generated its
    own rhythm," "memory internal to the tissue," "subharmonics generated by the tissue" all name
    the wrong object: what computes is the network, and "tissue" is the material it is made of.
    ⚠️ The material sense stays and is correct: "patient-derived tissue," "cultured neural
    tissue," "mouse and human tissue," "the same tissue yields different unit sets," "develops
    through the same self-organizing principles as cortical tissue." The test is whether the
    sentence attributes an action, a computation, or a memory to it; if so, say "network."
  These bans apply in prose, figure captions, and any text the reader sees. Internal code
  comments and variable names are exempt. — Hunter, this session ("campaign" 2026-09-02).
- [2026-09-02] Negative controls as hedges, not verbal qualifiers. Don't write "suggesting X" or
  "possibly Y"; instead run a concrete null (unit-count control, surrogate spikes, firing-rate
  replication) and let the null's failure carry the interpretive weight. "Confirming it originated
  in the tissue, not the model" rather than "suggesting it may originate in..." — voice analysis
  of results.tex (all mature sections use this pattern consistently).
- [2026-09-02] Closing sentences name the MECHANISM, not the result. Compare the cartpole closer
  ("Closed-loop control therefore drew on the network's existing dimensions") to the ASD section's
  recap ("Two unrelated ASD mutations... thus converge on a single quantitative shift"). The good
  ones say what the result MEANS; the weak ones restate it. Every subsection should end on a
  mechanistic or conceptual payoff, never a summary sentence. — voice analysis of results.tex.
- [2026-09-02] Every subsection opens with a conceptual anchor: 1-2 sentences of plain neuroscience
  framing before any method. "A foundational question in developmental neuroscience is..." The ASD
  section was the only one that skipped this, opening with a bare "We next asked whether" — the
  weakest opening in the paper. Every other section either opens with the biology or with a concrete
  callback to the prior figure's result. — voice analysis of results.tex.
- [2026-09-02] Stats always follow a claim, never lead. The interpretive sentence comes first,
  then the parenthetical stat. Never write a sentence whose only content is a number. — voice
  analysis of results.tex (consistent across all sections).
- [2026-09-02] ⛔ Banned "tissue" as the acting subject of a result (→ "network"); the material
  sense is fine. Fixed 3 sites: abstract.tex:6, introduction.tex:12, results.tex:166. — Hunter, this session.
- [2026-09-03] ⭐⭐ TWO PASS LEVELS — always clarify which one before editing. When asked to
  "go through" or "check" a paragraph, there are two very different jobs:
  **(1) Content-and-logic pass** (the default unless told otherwise): read each sentence cold as a
  naive reader. For each one ask: (a) does this sentence actually communicate its claim, or does the
  reader have to already know the result to parse it? (b) are the numbers present, sourced, and
  labeled (median vs mean, n, IQR)? (c) does the claim have a figure/stat backing it, or is it
  asserted bare? (d) is there a dangling referent ("each part", "this", "both") whose antecedent
  the reader lost? (e) does the sentence introduce something (a decomposition, a comparison, a
  control) that the paragraph then never follows up on? Report every gap BEFORE proposing rewrites,
  and look up missing numbers in the code/CSVs rather than asking Hunter what they are.
  **(2) Surface/grammar pass**: fix typos, punctuation, word bans, citation placement, and flow
  without changing the content or structure. Only do this when explicitly asked for a "grammar pass",
  "polish", or "light edit."
  If the request is ambiguous ("this sounds off", "can you fix this paragraph"), ask which pass
  before starting. Never default to a surface pass when the problem is content. — Hunter, this
  session (repeated pattern: content problems flagged only after Hunter pointed at individual
  sentences; a cold sentence-by-sentence content read would have caught them upfront).
- [2026-09-10] ⛔ When Hunter hands over citations "for" a section, they go ONLY in that
  section's prose. Never spread them into a figure caption or Methods, even if the paper looks
  like a protocol source; a citation's topical fit (e.g. "NGN2 = a different developmental
  timing") is his call, not a claim that the protocol was adapted from it. — Hunter, this session.
<!-- append new dated entries above this line -->
