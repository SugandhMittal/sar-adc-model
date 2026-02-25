# SAR-ADC Behavioral Model

A Python behavioral model of an N-bit **Successive Approximation Register ADC (SAR-ADC)** with non-ideality injection and standard performance characterization.

Built as a portfolio project to demonstrate analog IC design knowledge — the same analysis you'd do in Cadence Virtuoso post-layout simulation, implemented in Python and fully open-source.

---

## What it does

Models the complete SAR conversion algorithm and injects five real-world non-idealities:

| Non-ideality | Physical origin | Effect on performance |
|---|---|---|
| Comparator offset | Input-referred static offset Vos | Shifts transfer curve, code offset error |
| kT/C thermal noise | Noise frozen on CDAC sampling capacitor when switch opens | SNR floor set by √(kT/C) |
| Capacitor mismatch | Process variation in binary-weighted CDAC capacitors | DNL/INL errors, ENOB degradation |
| Clock jitter | Aperture uncertainty on the sampling instant | Noise floor that grows with input frequency |
| Reference noise | Thermal noise on the bandgap reference voltage | Random scaling error each conversion cycle |

Then computes standard ADC metrics from the output codes:
- **FFT spectrum** (dBFS)
- **SNDR** — Signal-to-Noise and Distortion Ratio
- **SFDR** — Spurious-Free Dynamic Range
- **ENOB** — Effective Number of Bits
- **DNL / INL** — Differential and Integral Non-Linearity (histogram method)

Also models **overdrive detection** — when the input exceeds the ADC range the algorithm saturates naturally at code 0 or 255 without artificial clipping, matching real hardware behaviour.

---

## Project structure

```
sar_adc/
├── adc_core.py         Core SAR-ADC behavioral model (ideal)
├── nonidealities.py    Five non-ideality models
├── characterize.py     FFT, SNDR, SFDR, ENOB, DNL, INL computation
├── demo.py             Run all models, print summary, generate plots
├── test.py             Unit tests — run before committing
└── README.md
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/SugandhMittal/sar-adc-model.git
cd sar-adc-model

# Install dependencies
pip install numpy matplotlib

# Run tests first
python test.py

# Run the demo
python demo.py
```

---

## Example output

Running `demo.py` compares the ideal ADC against each non-ideality and prints:

```
[Ideal]
=============================================
       SAR-ADC PERFORMANCE SUMMARY
=============================================
  SNDR          : +48.89 dBFS
  ENOB          :  7.83 bits
  SFDR          :  66.66 dBc
  DNL (peak)    :  0.496 LSB
  INL (peak)    :  0.498 LSB
=============================================

[Cap. Mismatch (0.5%)]
=============================================
  SNDR          : +48.22 dBFS
  ENOB          :  7.72 bits
  SFDR          :  59.69 dBc
  DNL (peak)    :  1.000 LSB
  INL (peak)    :  1.099 LSB
=============================================
```

And generates a characterization plot showing FFT spectrum, DNL, and INL for all six models side by side.

---

## Key concepts

**Why SAR?**
The SAR architecture is the dominant ADC topology for medium-speed, medium-resolution applications (8–16 bit, 1 kS/s – 100 MS/s). It is used in microcontrollers, sensor interfaces, and data acquisition systems. Understanding its non-idealities is fundamental to any analog IC design role.

**Why behavioral modeling?**
Transistor-level simulation in Cadence Virtuoso is accurate but slow and tool-locked. Behavioral models run in milliseconds, can be swept over thousands of Monte Carlo samples, and let you explore architecture trade-offs before committing to silicon. This is how ADC architects think about designs before writing a single schematic.

**Coherent sampling:**
The test signal uses a coherent frequency — an exact integer number of cycles fits within the FFT window. This eliminates spectral leakage, giving a clean single-bin fundamental tone and accurate SNDR measurement. The same requirement applies on real ADC test benches.

**kT/C noise floor:**
For a 1 pF sampling capacitor at 300 K:
`v_noise_rms = √(kT/C) = √(1.38e-23 × 300 / 1e-12) ≈ 64 μV RMS`
For a 1 V reference and 8-bit ADC, 1 LSB = 3.9 mV, so kT/C noise is well below 1 LSB at 1 pF. At 100 fF it becomes significant — this is the fundamental trade-off between capacitor size, power, and noise in scaled CMOS.

**Dynamic jitter model:**
Clock jitter noise is computed from the instantaneous signal slope using `np.gradient` rather than a worst-case constant. This means the jitter error is correctly small near signal peaks and maximum at zero crossings, matching the physics of aperture error.

---

## Roadmap (contributions welcome)

- [ ] Monte Carlo sweep: ENOB distribution over N mismatch samples
- [ ] ENOB vs. input frequency sweep (jitter limit plot)
- [ ] Delta-Sigma ADC model for comparison
- [ ] Jupyter notebook version with interactive widgets
- [ ] Export metrics to CSV for batch analysis

---

## Acknowledgements

Core behavioral model, non-ideality physics, and characterization algorithms written independently. Visualization script (`demo.py`) generated with AI assistance (Claude, Anthropic).

---

## Author

**Sugandh Mittal**
Erasmus Mundus Joint Master's in Microelectronics (RADMEP)
([linkedin.com/sugandhmittal](https://www.linkedin.com/in/sugandh-m-a75b97215/)) · ([github.com/SugandhMittal](https://github.com/SugandhMittal))
