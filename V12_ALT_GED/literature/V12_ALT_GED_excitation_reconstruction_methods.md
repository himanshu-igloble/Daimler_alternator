# V12 — Reverse-Engineering the Alternator Excitation: Methods Survey

**Question:** what are the ways we can reverse-engineer / reconstruct the **continuous alternator field-excitation current and PWM duty cycle** when those quantities live only on the in-alternator LIN bus (regulator `RDC`/`RMC` registers) and are absent from our 6-signal, ~5 s (0.2 Hz) CAN feed (`VSI, RPM, ANR, CSP, SMA, GED`)?

**Method:** internet deep-research harness — 6 search angles, 25 sources fetched, 109 claims extracted, **25 claims put through 3-vote adversarial verification (24 confirmed, 1 killed)**, synthesized to 8 findings. Date 2026-06-26. Confidence levels and the proxy-vs-reconstruction distinction are carried through; pages that were 403/JS-blocked on direct fetch are flagged as corroborated-not-first-hand-read.

---

## Bottom line (verified)

> **True continuous reconstruction of field current or PWM duty cycle from our 6 CAN signals at 5 s is NOT achievable.** Every method that genuinely recovers field current needs hardware and bandwidth we don't have (a real current sensor + kHz sampling), and the one "free" analytical route — invert bus voltage → field current — is **double-blocked**: the regulator deliberately holds VSI flat (~28 V) so the bus voltage carries almost no field-current information, and the Lundell magnetic circuit is saturated so the relationship is strongly nonlinear. **What IS feasible is an honest electrical-LOAD / regulation-effort HEALTH PROXY — not the excitation value itself.** The single strongest feasible avenue uses a channel we already have: **engine torque (ANR)**.

This *confirms and sharpens* V12's "sensor-blocked, not method-limited" conclusion, and it identifies one concrete proxy V12 did **not** build (the torque-residual load proxy; V12 Phase-3B built only the VSI-regulation proxy).

---

## Method-by-method assessment

### (1) Physics state observers (Kalman/EKF/UKF, Luenberger, sliding-mode) — TRUE reconstruction, but INFEASIBLE
- **What it recovers:** the actual rotor field current (true reconstruction).
- **Verdict:** **infeasible at our constraints.** Observability analysis shows every observable configuration of a wound-rotor/claw-pole synchronous machine contains **≥1 real current sensor** (on the rotor field winding or a stator phase); **terminal voltage + shaft speed alone leave the field current unobservable** (C=0 → rank 0). Observers also run at the switching cadence (~10 kHz, Ts≈100 µs) — ~5 orders of magnitude faster than 0.2 Hz.
- **Source:** Eull, Parker & Preindl, *Wound-rotor synchronous machine current estimation using a linear Luenberger observer*, IEEE ITEC 2022 (DOI 10.1109/itec53557.2022.9814015). Vote **3-0**. Confidence **High**. *(Read in full; the only first-hand-read primary source.)*
- **Caveat:** that paper is an inverter-fed EV traction drive with **known applied voltages** — our self-rectified alternator with an internal LIN PWM regulator and only post-rectifier bus voltage is **even less observable**, reinforcing the negative result.

### (2) Analytical machine-model back-calculation (VSI + speed → field current) — BLOCKED → proxy only
- **What it recovers (in principle):** field current, by inverting a Lundell output model.
- **Forward model exists:** output voltage rises with the field-current × speed product (`E ≈ k·ω·φ`); high-fidelity magnetic-equivalent-circuit (MEC) and electromagnetic models capture saturation and saliency; analytical equations for field winding / field current / excitation losses are published. But these are **design-stage forward models (field → output), not runtime estimators.**
  - Sources: *Model of Automotive Alternator Output Voltage with Rectifier and Voltage Regulator* (ResearchGate 385188775); Ostovic et al., MEC of a Lundell alternator (IEEE Trans IA 35(4):825-830); *Electromagnetic model for the Lundell alternator with switched-mode rectifier* (IEEE Xplore 4736651); *Excitation current control of a claw-pole automotive alternator* (IEEE 6713044). Votes **3-0**. Confidence **High** *(corroborated via search + verbatim-quote match + confirmed IEEE IDs; several pages 403-blocked, not first-hand-read).*
- **Why inversion is blocked (the key distinction):** (a) the regulator **maintains constant bus voltage despite varying speed and load**, so VSI→field-current is non-invertible without a load model; (b) the magnetic circuit is **saturated at rated field current**, so any inversion must embed a Newton-Raphson saturation model and a naive linear inversion fails. Source: IntechOpen ch.38166 (Ivankovic et al. 2012); Ostovic MEC. Vote **3-0**. Confidence **High**.
- **Verdict:** VSI-based methods are an **honest health proxy, not true reconstruction.** *(This is exactly what V12 Phase-3B is — and the research validates the framing.)*
- **The target law (for reference):** field-PWM **duty = f(speed↓, load↑)** — duty decreases as rotor speed rises and increases proportionally to load current. Invertible in principle but needs **load current**, which we don't have. Source: ResearchGate 385188775 + ST L9409 / NXP MC33092A datasheets. Vote **3-0**.

### (3) Engine-torque (ANR) residual → alternator electrical-load proxy — STRONGEST FEASIBLE AVENUE
- **What it recovers:** alternator **electrical load / output current** (an honest proxy) — **not** duty cycle or field current.
- **Forward physics is rock-solid:** alternator drag torque on the accessory drive obeys **`T = V·i / (η·ω)`** — electrical power ÷ (efficiency × shaft speed). At ~constant V and η (e.g., idle / short windows) this reduces to **`T ≈ i/ω`**, so **at a fixed RPM an ANR residual maps monotonically to alternator output current.** Monotonicity is robust even if η varies (`T=(V·i+losses)/ω`, copper loss ∝ i², rectifier loss ∝ i).
  - Sources (primary OEM patents, verbatim-confirmed): **US7,283,899 B1 (Remy, 2007)** `T=V·i/(n·ω)`; **US9,126,580 B2 (Ford, 2015)** "engine torque as a function of alternator output (watts) and alternator speed"; **US10,752,188 B2 (CNH/Blue Leaf)** Eq. 6 parasitic loss = P_alt/η(ω,I) → drag = P_elec/(η·ω). Votes **3-0**. Confidence **High** *(patent pages 403-blocked on USPTO; corroborated via Google Patents mirrors + quote match).*
- **Why it's the best fit for us:** we **already have ANR (engine torque) and RPM (speed)** — no new signal required.
- **Honest limits:**
  - It is a **load/current proxy, not duty/field.** Alternator torque is a 2-D surface `Ta=f(ω, duty)` modified by **temperature**, so duty is **not uniquely invertible** from torque without also knowing speed and temperature. Source: **US9,020,721 B2 (FCA)** — duty cycle is a *required input* to the torque calibration, not an output. Vote **3-0**.
  - Published *inverse* torque methods need bandwidth we lack: the academic instantaneous load-torque estimator uses a **sliding-mode observer on per-cycle crankshaft angular velocity + in-cylinder pressure from block vibration** (kHz / crank-angle-resolved). Source: **SAE 980795** (Azzoni/Moro/Ponti/Rizzoni 1998). Vote **3-0**.
  - **Magnitude problem (feasibility-in-practice):** alternator drag is only **~1–5 Nm** within total engine torque; isolating it from ANR at 5 s amid other parasitic loads is a *separate, non-trivial estimation problem the physics supports but the sources do not demonstrate at this cadence.*

### (4) Grey-box / physics-informed ML surrogate — valid proxy route, weakly sourced for automotive
- **What it recovers:** a regulation-effort / load **health index** (proxy), by fusing a first-principles model with a data-driven model.
- **Verdict:** **methodologically valid** — soft (virtual) sensors exist precisely to estimate states that are "difficult or impossible to measure through hardware due to delays, technological limitations or cost"; grey-box models give higher accuracy than pure physics and more interpretability than pure ML. Source: Ahmad et al., *Gray-box Soft Sensors in Process Industry*, Processes (MDPI) 8(2):243, 2020. Vote **3-0**. Confidence **Medium** — the only surviving source is **process-industry, not automotive-alternator**; relevance is by analogy and accuracy depends on the embedded physics being right.

### (5) Minimal-instrumentation + transfer learning (LIN-tap a few reference trucks → map to fleet) — UNADDRESSED by sources
- **What it would recover:** true `RDC`/`RMC` ground truth on a few trucks → a transferable fleet proxy (digital-twin calibration).
- **Verdict:** the only route that could anchor a *true* excitation value at scale, but **no surviving source addresses it directly** — the recommendation is an **engineering inference**, not a cited result. Open question: cross-truck transfer error achievable at n≈25.

### (6) Electrical-signature methods (DC-bus ripple, load-dump, coast-down) — INFEASIBLE at 0.2 Hz
- **Verdict:** **ruled out by sampling.** These signatures live at hundreds of Hz to kHz; at 0.2 Hz they are fully aliased (Nyquist). Directly evidenced for the kHz methods (observers 10 kHz; in-cycle torque per-combustion); the ripple/load-dump/coast-down extension is a **sound Nyquist inference** (no dedicated ripple claim survived verification). Confidence **Medium** on breadth; the sampling math itself is certain. Sources: Eull ITEC 2022; SAE 980795.

### (7) SAE/IEEE/automotive virtual-sensor literature
Covered across (1)–(4) above. The relevant primary literature is the observer paper (Eull), the Lundell analytical/MEC models, the OEM torque patents (Remy/Ford/FCA/CNH), and the soft-sensor review (MDPI). LIN-regulator datasheets (Infineon TLE8881-2, NXP TC80310/AN4289, ST L9409) confirm the duty-cycle/field-current registers exist **inside** the alternator — i.e., the quantities are real but off-CAN.

---

## Refuted / unresolved (transparency)
- **REFUTED (vote 1-2):** the claim that an **over-run (decoupler/OAD) clutch zeroes alternator torque** whenever rotor speed exceeds engine speed — which would be a *decisive confounder* for the ANR torque-residual proxy during decelerations — **did not survive verification.** Whether a decoupler pulley nulls the torque proxy is therefore **unresolved**, not established. This must be checked against our actual trucks before relying on the torque proxy.

## Honest caveats
- **Proxy ≠ reconstruction.** No surviving method reconstructs the true continuous field current / duty cycle from our 6 signals at 5 s. Every feasible route is a health proxy.
- **Source access:** only the Eull observer PDF was read in full first-hand; the patents and several IEEE/IntechOpen pages were 403/JS-blocked and corroborated via independent search, verbatim-quote matching, and confirmed DOIs/patent IDs — treat exact wording as corroborated, not first-hand-read.
- **Machine-class analogy:** the only true field-current observer evidence is an EV traction drive with known applied voltages, not a self-rectified alternator (which is *less* observable).
- **Grey-box source is off-domain** (process industry) → medium confidence.

---

## Recommended ranked path for OUR constraints

1. **Build the engine-torque (ANR-at-fixed-RPM) residual → alternator electrical-load proxy.** Most feasible — uses signals we already have (ANR, RPM). Honest "alternator electrical load" health indicator, **not** duty/field. *First validate separability of the ~1–5 Nm alternator drag from total ANR at 5 s, and check the OAD/decoupler confounder on our trucks.* This is the natural V12 follow-on (a sibling to the existing Phase-3B VSI-regulation proxy, and likely complementary to it).
2. **Wrap a grey-box / physics-informed surrogate** that fuses `T≈i/ω` and the VSI-regulation residual with a small data-driven model → a single regulation-effort health index. Anchor/gate it with the **GED=2 "disturbance" enum** (the only excitation-state signal already on CAN) as a coarse native label.
3. **Only if a true excitation *value* is ever required:** instrument a handful of reference trucks with **LIN taps or field-current shunts** to capture ground-truth `RDC`/`RMC`, and learn a transferable mapping (digital-twin / domain adaptation). This is also exactly the **V12 sensor-gap recommendation** (add LIN `RMC`/`RDC` + SPN 115 + alt temp), now reframed: instrument a few, transfer to the fleet.
4. **Reject** (don't spend effort): observer-based field-current reconstruction, VSI→field-current analytical inversion, and all kHz electrical-signature methods (ripple / load-dump / coast-down) — infeasible at 0.2 Hz with our signals.

## Open questions / next steps
- Is the ~1–5 Nm alternator drag actually **separable** from total ANR at 5 s with 6 signals (feasibility, not just principle)? Does an OAD/decoupler pulley confound it on our trucks?
- Can **GED=2** anchor/validate/gate a torque-based or grey-box regulation-effort proxy?
- What cross-truck **transfer error** is achievable at n≈25 for the reference-truck-instrumentation route?
- What real accuracy can a grey-box alternator regulation-effort proxy reach (no automotive benchmark exists yet)?

## Sources (verified set)
- Eull, Parker, Preindl — linear Luenberger observer, WRSM current estimation, IEEE ITEC 2022 — strathprints.strath.ac.uk/81564
- *Model of Automotive Alternator Output Voltage with Rectifier and Voltage Regulator* — ResearchGate 385188775
- IntechOpen ch.38166 — Ivankovic et al. 2012 (regulator holds constant voltage)
- Ostovic et al. — MEC performance computation of a Lundell alternator — academia.edu/11114070; IEEE Trans IA 35(4)
- *Electromagnetic model for the Lundell alternator with switched-mode rectifier* — ResearchGate 251869119 / IEEE 4736651
- *Excitation current control of a claw-pole automotive alternator* — IEEE 6713044
- **US7,283,899 B1 (Remy)** `T=V·i/(n·ω)`; **US9,126,580 B2 (Ford)**; **US9,020,721 B2 (FCA)** torque calibration vs duty; **US10,752,188 B2 (CNH)** parasitic-loss = P/η
- SAE 980795 — Azzoni/Moro/Ponti/Rizzoni, sliding-mode load-torque observer
- Ahmad et al. — *Gray-box Soft Sensors in Process Industry*, Processes (MDPI) 8(2):243, 2020
- LIN-regulator datasheets: Infineon TLE8881-2; NXP TC80310 / AN4289; ST L9409 (confirm `RDC`/`RMC` exist inside the alternator)

*Full machine-readable result (claims, votes, evidence, all 25 sources): the deep-research workflow output. This document is the curated synthesis.*
