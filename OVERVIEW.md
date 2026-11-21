# Why Fluency Isn't Truth: A Plain-Language Guide to the Paper & Experiments

**Companion overview for:** *Truth as a Structural Invariant: A Sheaf-Bayesian Epistemology for Superintelligence*  
**Author:** Rafael Espericueta ([ShimmeringVoid.org](https://ShimmeringVoid.org))

---

## The Starting Puzzle: Why Articulate AI Still Hallucinates

Modern large language models (LLMs) are trained on a remarkably simple objective: given a string of words, predict what word is most likely to come next across a massive archive of human text.

That training recipe produces astonishing fluency, but fluency and truth are not the same thing. As Léon Bottou and Bernhard Schölkopf put it, standard language models behave like **"fiction machines"** optimized for narrative plausibility. In his 2026 essay *Against All Likelihood*, David Lopez-Paz diagnoses the root cause: **statistical likelihood measures how common or familiar a pattern is inside one historical dataset, whereas truth is about what holds up when conditions change.**

When all the world's text is poured into one giant training bucket, two things go wrong:
1. **Popularity masquerades as reliability.** If a misconception is repeated on ten thousand web pages and a careful correction appears in one technical monograph, predicting the "most likely" continuation favors the crowd and the most familiar writing style rather than the most reliable source.
2. **What sources *say* gets confused with how the world *works*.** Modeling how people talk about a topic is not the same as modeling the underlying mechanism that makes a statement true or false.

Shortcuts make this worse. An image model trained on vacation photos quickly learns to classify "cow" by checking for green grass pixels; a language model learns surface shortcuts and memorizes exceptions rather than learning exact multi-digit arithmetic. Inside the training distribution, shortcuts look great. Step outside it, and they fall apart.

---

## The Core Idea: Truth as What Stays Invariant

How does science tell reality apart from illusions, coordinate quirks, and popular rumors? Across physics and the philosophy of science—from Henri Poincaré, Hermann Weyl, and Robert Nozick to Lopez-Paz—the answer is **structural invariance**: *what is real is what remains consistent when you change your point of view.*

* Change your measurement units or reword a question in a flattering tone, and a genuine law does not budge.
* Approach the same phenomenon from two genuinely independent disciplines—say, deriving the ideal gas law from microscopic collisions in kinetic theory and from macroscopic heat engines in thermodynamics—and both routes converge on the same answer.
* Translate local measurements across overlapping scientific domains, and true descriptions fit together seamlessly around closed loops, whereas fabrications and calibration errors leave detectable seams.

This paper asks a concrete question: **How do you build that scientific standard of truth into a mathematical and algorithmic blueprint for advanced AI systems, using Bayesian probability and network topology?**

---

## The Three-Layer Architecture in Plain English

Instead of treating all text and reasoning as one flat stream of tokens, the paper separates truth-seeking into **three cooperating layers**:

### Layer 1: Counting Independent Sources, Not Echoes (Provenance & Calibration)
In everyday life, one person shouting the same claim a hundred times is still just one witness. On the internet—and inside AI training sets—the echo-chamber problem is subtler: fifty articles might all paraphrase the same underlying press release or copy the same flawed study, creating the illusion of fifty independent witnesses.

Layer 1 models source copying mathematically:
* **Single-source saturation:** Repeating a claim inside a single channel increases confidence slightly (by ruling out a slip of the tongue) and then hits a hard ceiling. No amount of repetition from one source can overcome the chance that the source itself is mistaken or biased.
* **The "too good to be true" effect:** When sources *are* known to be independent, agreement compounds rapidly. When we *don't* know who copied whom, unverified unanimity grows far more slowly—and under a plausible chance of a hidden common origin, suspiciously unanimous agreement can actually peak and dip back down toward the weight of a single witness.
* **Why verification matters:** Rapid corroboration returns when claims carry **costly, verifiable derivations** (such as formal proofs or reproducible checks tied to distinct evidence) that cannot be faked just by paraphrasing sentences.
* **Capping overconfidence:** Layer 1 also places a ceiling on how much precision any single source is allowed to claim, preventing one "infinitely confident" voice from dominating downstream reasoning.

### Layer 2: Real Consilience vs. Word Games (Reasoning & Knowing When to Abstain)
Asking an AI the same question ten times and taking a majority vote is a popular trick, yet not all variations of a prompt mean the same thing:
* **Nuisance framing shifts**—translating a question into another language, shuffling the order of options, or adding a sycophantic hint ("My professor thinks the answer is B, don't you?")—should *not* change the underlying reasoning or conclusion.
* **Substantive epistemic angles**—testing a claim through genuinely different sub-theories, counterfactuals, or disciplinary models—*should* follow distinct valid reasoning paths that converge on the same answer (what the 19th-century philosopher William Whewell called **consilience**).

Layer 2 trains reasoning policies using **exploratory Generative Flow Networks (GFlowNets)**—an off-policy learning method that deliberately explores diverse reasoning paths instead of collapsing onto the single most rehearsed habit. It filters out irrelevant angles, uses Layer 1 to keep paraphrased duplicate prompts from double-counting as consilience, and applies **Chow's abstention rule**: when rival explanations remain unresolved and confidence falls short of a safety threshold, the system explicitly says **"abstain"** rather than guessing.

### Layer 3: Stitching Domains Together Without Getting Hijacked (Knowledge Sheaves)
Different fields of knowledge describe overlapping parts of the world in their own local coordinate systems. Mathematicians have a clean tool for describing local views connected by pairwise translation bridges: a **cellular sheaf**. When local claims agree across every bridge, they form a coherent global picture; when they clash, the discrepancy shows up geometrically.

Real-world knowledge integration faces two simultaneous hazards: **imperfect translation bridges** between domains and **faulty or colluding sources**.
* Standard averaging (Gaussian pooling) is fragile: a single source with wildly inflated confidence can drag the entire network toward a falsehood.
* Heavy-tailed ($t$-distribution) statistics naturally downweight extreme outliers, yet on a network with slightly noisy translation bridges, a subtle trap appears: far from the origin, tiny bridge-alignment errors make *honest* sources look slightly inconsistent, causing a naive robust filter to distrust everyone and collapse toward zero.
* Layer 3 solves this with a **staged robust filter**: first fitting a geometric spatial median across the network's approximate invariant subspace (weighted by Layer 1's source-independence scores), and then running heavy-tailed expectation–maximization. Alongside vertex filtering, a **two-step topological diagnostic** (using discrete Hodge theory) inspects loops of bridges to distinguish a local broken translation link from a global contradiction circulating around a cycle of domains.

---

## What the Four Python Experiments Actually Test

A central rule of this project is **zero hand-typed numbers**: running `simulate_sheaf_bayes_v8_6.py` executes four synthetic testbeds from scratch (`seed=42`) and writes all 438 numbers and 20 table bodies directly into the LaTeX manuscript, alongside the three figures. Just as importantly, the experiments map out **both where the methods succeed and the exact boundaries where they break**.

### Experiment 1: Echo Chambers, Paraphrase Attacks, and Forged Verifiers
* **What it simulates:** Millions of noisy utterances inside single channels, unanimous multi-source agreement under unknown provenance, and adversarial "paraphrase attacks" where 6 cloned outlets reword a falsehood to outvote 4 genuinely independent truth-tellers.
* **What we see:** Repeating a claim within one channel flatlines at a finite Bayes-factor ceiling ($6.14$ in the test setup), whereas naive models that ignore source copying either decay paradoxically or blow up to infinity. Paraphrased clones easily fool surface text-similarity checks, while requiring verified derivations holds posterior confidence in the truth above $98\%$—*unless* the verifier itself is unsound and lets clones forge distinct proof IDs, which flips confidence to the falsehood.

### Experiment 2: Surviving Colluders and Clone Majorities on a Noisy Network
* **What it simulates:** A 20-domain network translating 2D state vectors across 30 noisy rotation bridges. Up to $k$ domains collude around a noiseless false claim while asserting $500\times$ base precision (simulating an attacker who bypasses calibration).
* **What we see:**
  * **Near the origin:** Standard Gaussian pooling is pulled off course by even a single $1000\times$-confident rogue domain, whereas warm-started heavy-tailed sheaf filtering resists capture by up to $9$ out of $20$ colluders ($45\%$, matching the theoretical $\lfloor(N-1)/2\rfloor$ breakdown ceiling).
  * **Far from the origin:** Uncentred heavy-tailed filtering suffers catastrophic shrinkage toward zero once bridge-map discrepancies push honest data into the distribution tails. Two-stage spatial-median centering stops that collapse completely, resisting capture once a false claim's offset clears roughly $10$–$15\times$ the bridge-noise scale.
  * **Defeating a $60\%$ clone majority with Layer 1:** No purely geometric filter can outvote $12$ out of $20$ sources ($60\%$) on equal footing. When Layer 1 computes exact provenance weights from source signatures, $12$ unverified clones are downweighted enough that even a small handful of verified honest domains keeps total honest weight ahead of faulty weight ($S_{\mathrm{fault}} < S_{\mathrm{hon}}$) and prevents capture.
  * **A three-stage map-aware comparison:** Giving the filter the known bridge-noise scale loosens stiff edges far from the origin—restoring single-domain accuracy on clean data and lowering the capture margin to about $5\times$—at the honest trade-off of leaving the faulty vertices themselves uncorrected at moderate offsets.

### Experiment 3: Discovering Spacetime Relativity & Diagnosing Broken Loops
* **What it simulates:** First, three moving observers exchanging clock and position readings of 20 spacetime events across velocities from $0.05c$ to $0.73c$, comparing Newtonian/Galilean shear maps against Einstein's Lorentz boosts without hand-tuning the sheaf coupling strength. Second, a 6-node, 12-edge triangulated ring (annulus) injected with either local single-bridge errors or global loop contradictions.
* **What we see:** Once relative speeds exceed clock noise, Galilean transformations fail to close around the 3-observer loop (missing the relativity of simultaneity), and dial-free Bayesian evidence strongly selects the Lorentz sheaf—while coarse clocks recover single-event Newtonian indistinguishability until enough events accumulate. On the ring complex, the two-step topological rule cleanly separates local triangle faults from global loop obstructions, with exact analytical decision and mislabeling rates.

### Experiment 4: Escaping Reasoning Ruts on a Symbolic Consilience Network
* **What it simulates:** Twelve candidate theories (combining invariant, shortcut, and entangled features with linear and nonlinear modular arithmetic laws) evaluated across 120 reasoning trajectories and 5 epistemic angles, including a dead out-of-scope angle and cloned sycophantic prompts.
* **What we see:** Policies trained purely *on-policy* frequently get trapped in suboptimal reasoning modes ($30\%$–$64\%$ failure rate from peaked initializations), whereas mixing in *off-policy* exploratory sampling drives total-variation error near zero. Scope-gating ignores dead angles, verified signatures prevent $6\times$ paraphrased shortcut prompts from crossing the commitment threshold (triggering a safe `ABSTAIN`), and adding a genuine disambiguating angle lets the system commit accurately to the true invariant law.

---

## Honest Scope & Limitations

In keeping with its own theme, the paper is explicit about what it proves and what remains open:
* **A mathematical and synthetic blueprint, not a trained LLM:** Every experiment runs on small, transparent synthetic testbeds ($20$ domains, $60$ symbolic states) where ground-truth mechanisms and failure boundaries can be inspected exactly. Scaling to millions of claims requires amortized provenance samplers, sparse eigensolvers, and learned exploration policies.
* **Garbage-in verifier risk:** Both Layer 1 and Layer 2 rely on having a *sound* formal checker (such as Lean/Coq proof verification, code execution sandboxes, or validated physical checks). If an attacker can mint distinct "verified" IDs for copied or invalid arguments, deduplication fails.
* **Theoretical vs. benchmarked modules:** Active experimental design (choosing which real-world test to run next) and continual filtering over time are formulated mathematically in Section 5 as part of the overall architecture, awaiting end-to-end empirical testing in future work.

---

## Where to Dive In

* **Want the full mathematical treatment and all 20 result tables?** Read [`sheaf_bayesian_epistemology_v8_7.pdf`](sheaf_bayesian_epistemology_v8_7.pdf).
* **Want to inspect or run the experiments yourself?** Everything lives in a single, dependency-light Python file, [`simulate_sheaf_bayes_v8_6.py`](simulate_sheaf_bayes_v8_6.py), with step-by-step run instructions in [`README.md`](README.md).