# V12 — Referenced Research Papers (excitation reverse-engineering survey)

Reference list + downloaded PDFs supporting `../V12_ALT_GED_excitation_reconstruction_methods.md`.
US patents are US-government **public-domain**; the MDPI paper is **open-access (CC BY 4.0)** — both freely redistributable. PDFs in this folder were downloaded for the two findings flagged for retrieval (engine-torque proxy; grey-box ML). All other verified sources are listed with stable URLs/DOIs.

Downloaded 2026-06-26 (Google Patents PDF storage for patents; `res.mdpi.com` file host for MDPI — the `www.mdpi.com/.../pdf` endpoint is Akamai bot-blocked).

---

## Downloaded PDFs (this folder)

### Finding ✅ — Engine-torque (ANR) residual → alternator electrical-load proxy  ·  `T = V·i / (η·ω)`
| # | PDF | Citation | What it supports |
|---|-----|----------|------------------|
| 1 | `US7283899B1_Remy_alternator_drag_torque.pdf` (1.24 MB) | **US 7,283,899 B1**, Remy International, granted 2007 | Core relation `T = V·i/(n·ω)` (torque = output-voltage × output-current / (efficiency × rotor-speed)); at idle V,n ≈ const → `T ≈ i/ω` |
| 2 | `US9126580B2_Ford_engine_torque_vs_alternator_output.pdf` (1.64 MB) | **US 9,126,580 B2**, Ford Global Technologies, granted 2015 | "engine torque as a function of alternator output (e.g., watts) and alternator speed" |
| 3 | `US10752188B2_CNH_alternator_parasitic_loss.pdf` (2.48 MB) | **US 10,752,188 B2**, CNH Industrial America / Blue Leaf | Eq. 6: parasitic mechanical loss `PL_alt = P_alt(e)/η(ω,I)` → drag torque = electrical power / (η·ω) |

### Finding ✅ — Grey-box / physics-informed ML surrogate (regulation-effort health index)
| # | PDF | Citation | What it supports |
|---|-----|----------|------------------|
| 4 | `MDPI_Processes_2020_8-2-243_graybox_soft_sensors.pdf` (540 KB) | **Ahmad, Ayub, Kano & Cheema (2020)**, *Gray-box Soft Sensors in Process Industry: Current Practice, and Future Prospects in Era of Big Data*, **Processes 8(2):243**, DOI **10.3390/pr8020243** | Soft sensors estimate states "difficult or impossible to measure through hardware"; grey-box = physics (white-box) + data (black-box) → higher accuracy than pure physics, more interpretable than pure ML. *(Process-industry domain; relevant to alternators by methodological analogy — medium confidence.)* |

---

## Other verified sources (referenced, not downloaded)

**Observers (Category 1):**
- Eull, Parker & Preindl, *Wound-rotor synchronous machine current estimation using a linear Luenberger observer*, IEEE ITEC 2022 — DOI 10.1109/itec53557.2022.9814015 — https://strathprints.strath.ac.uk/81564/ — *(field current unobservable from voltage+speed alone; needs ≥1 current sensor + kHz sampling)*

**Lundell/claw-pole analytical + regulator models (Category 2):**
- *Model of Automotive Alternator Output Voltage with Rectifier and Voltage Regulator* — ResearchGate 385188775
- Ivanković et al. (2012), IntechOpen chapter 38166 — https://www.intechopen.com/chapters/38166 — *(regulator holds bus voltage constant → VSI→field non-invertible)*
- Ostović et al., *MEC-based performance computation of a Lundell alternator*, IEEE Trans. Ind. Appl. 35(4):825-830 — academia.edu/11114070 — *(magnetic circuit saturated at rated field current)*
- *An electromagnetic model for the Lundell alternator with switched-mode rectifier* — IEEE Xplore 4736651 / ResearchGate 251869119
- *Excitation current control of a claw-pole automotive alternator* — IEEE Xplore 6713044

**Engine-torque caveats (Category 3):**
- **US 9,020,721 B2** (FCA) — alternator torque = calibration surface f(speed, duty-cycle) modified by temperature; **duty cycle is a required input, not invertible from torque** — https://patents.google.com/patent/US9020721B2/en
- **SAE 980795** — Azzoni, Moro, Ponti, Rizzoni (1998), instantaneous load-torque via sliding-mode observer on per-cycle crankshaft angular velocity — https://www.sae.org/publications/technical-papers/content/980795/ — *(needs kHz crank-resolved data)*

**LIN voltage-regulator datasheets (confirm `RDC` duty-cycle / `RMC` field-current registers exist inside the alternator):**
- Infineon TLE8881-2 — https://www.infineon.com/dgdl/Infineon-TLE8881-2-DataSheet-v01_00-EN.pdf
- NXP TC80310 / AN4289 — https://www.nxp.com/docs/en/data-sheet/TC80310.pdf · https://www.nxp.com/docs/en/application-note/AN4289.pdf
- ST L9409 (referenced via the analytical sources)

---

## BibTeX (downloaded items)

```bibtex
@patent{remy_us7283899b1,
  title  = {Method and apparatus for determining the torque of an alternator},
  number = {US7283899B1}, assignee = {Remy International}, year = {2007},
  note   = {T = V*i/(n*omega); at idle T ~ i/omega}}

@patent{ford_us9126580b2,
  title  = {Method and system for engine torque control},
  number = {US9126580B2}, assignee = {Ford Global Technologies}, year = {2015},
  note   = {Engine torque as a function of alternator output (watts) and alternator speed}}

@patent{cnh_us10752188b2,
  title  = {Electrical power system for a work vehicle (alternator parasitic loss)},
  number = {US10752188B2}, assignee = {CNH Industrial America / Blue Leaf IP},
  note   = {Parasitic mechanical loss = electrical power / efficiency(omega, I)}}

@article{ahmad2020graybox,
  title   = {Gray-box Soft Sensors in Process Industry: Current Practice, and Future Prospects in Era of Big Data},
  author  = {Ahmad, Iftikhar and Ayub, Aimal and Kano, Manabu and Cheema, Izzat Iqbal},
  journal = {Processes}, volume = {8}, number = {2}, pages = {243}, year = {2020},
  doi     = {10.3390/pr8020243}, publisher = {MDPI}}
```
