# Data card

> Document every dataset used. Required even for synthetic data — it forces
> you to record how it was generated and why.

## Dataset name

- **Source:** <url / generator script>
- **License:** <e.g., CC-BY-4.0>
- **Size:** <samples, modalities, total bytes>
- **Splits:** how train / val / test are constructed; whether splits are
  deterministic; any group-level constraints (e.g., subject-wise split).
- **Preprocessing:** what transformations are applied at load time vs. at
  training time. Where they live in code.
- **Known issues / biases:** label noise, distribution skew, leakage risks.
- **Reproduction:** exact command(s) to obtain or regenerate the data.

(Repeat the section above for each dataset.)
