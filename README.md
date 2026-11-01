# otoium

A tool for automatically generating `oto.ini` from WAV recordings. Supports Japanese UTAU recording styles: CV (Consonant-Vowel), VCV (Vowel-Consonant-Vowel), and CVVC.

## Usage

```bash
pip install -r requirements.txt

python run_otoium.py <folder>                        # Accuracy: high / Mode: auto
python run_otoium.py <folder> --mode vcv --accuracy max
python run_otoium.py <folder> --mode cvvc --vcv-style vowel
python run_otoium.py <folder> --script script.txt --jobs 4 --textgrid

python run_otoium.py                                  # Launch GUI (Tkinter)
```

Main options:

- `--mode auto|cv|vcv|cvvc`
- `--accuracy fast|high|max|ultra|custom`
- `--iters N`
- `--no-tune`
- `--no-rescue`
- `--no-net`
- `--no-adapt`
- `--no-auto-thresholds`
- `--no-refine`
- `--jobs N`
- `--merge`
- `--prefix`
- `--suffix`
- `--head-prefix`
- `--tail`
- `--encoding`
- `--strict`
- `--labels`
- `--textgrid`
- `--model x.npz`
- `--dry-run`
- `--no-backup`
- `--no-cache`

## Accuracy Presets

| Preset | Gaussian Training | Auto-Tuning | NN (Epochs × Ensembles) | Adaptation | Retraining | High-Confidence Retraining |
|---|---:|---|---:|---:|---:|---:|
| fast | 3 | None | None | None | 0 | 0 |
| high | 5 | 1 round | 10 × 1 | 1 | 2 | 0 |
| max | 8 | 2 rounds | 14 × 2 | 2 | 3 | 1 |
| ultra | 12 | 3 rounds (fine-grained search) | 20 × 3 | 3 | 4 | 2 |
| custom | `--iters` | `--tune-rounds` | `--net-epochs` / `--net-ensemble` | `--adapt-rounds` | `--post-iters` | `--confident-rounds` |

These parameters can also override preset values without using `custom`.

For example:

```bash
python run_otoium.py <folder> --accuracy max --net-ensemble 4
```

The system also learns phoneme-specific duration distributions to adapt to differences in speaking and recording speed.

### Script File Format

One recording per line:

```text
filename.wav=text
```

Tab-separated and space-separated formats are also supported.

## Processing Pipeline

1. **Feature Extraction**
   - DC offset removal
   - Low-frequency filtering
   - Noise floor normalization
   - MFCC and delta features
   - Voicing probability
   - Approximate F1/F2 formant features
   - Fine-grained 1 ms envelope analysis

   Extracted features are cached in `.otoium_cache`.

2. **Self-Supervised Acoustic Model Training**
   - Gaussian models with Viterbi re-estimation, outlier file exclusion, weighted training, and convergence detection.
   - NumPy-based MLP frame classifier using contextual frames.
   - Alignment results are used as training targets, and neural network likelihoods are incorporated into decoding.

3. **Automatic Parameter Tuning**

   Optimizes boundary weights, duration weights, pause penalties, and decoder penalties by maximizing validation scores.

   Tuning is performed only on the alignment side. Validation data is excluded from tuning to prevent self-evaluation bias.

3b. **Automatic Threshold Calibration**

   Determines thresholds for boundary confidence, vowel consistency, and consonant consistency based on the distribution of the recording data.

4. **Recording-Specific Adaptation and Forced Alignment**
   - Adapts the model to individual recordings.
   - Performs forced alignment using duration-constrained semi-Markov Viterbi decoding.
   - Refines phoneme boundaries with 1 ms precision.

   Validation is performed using the model before recording-specific adaptation.

5. **Independent Validation**

   Uses a free decoder without script input to independently verify:
   - Vowel consistency
   - Consonant presence
   - Number of vowel nuclei
   - Boundary confidence

6. **Automatic Recovery of Failed Recordings**
   - Re-analyzes failed recordings with modified parameters.
   - Reduces trailing morae when a recording is shorter than its corresponding script.
   - Recordings that still fail validation are excluded from output.

7. **Alias Generation and oto Parameter Calculation**

## Output Files

| File | Description |
|---|---|
| `oto.ini` | Generated oto configuration |
| `oto_review.txt` | Recordings requiring manual review |
| `otoium_report.json` | Detailed processing report |
| `otoium_report.txt` | Human-readable processing report |
| `otoium_model.npz` | Trained acoustic model |

The default encoding for `oto.ini` is CP932.

Existing `oto.ini` files are backed up as `.bak` before being overwritten.

### Processing Status

| Status | Behavior |
|---|---|
| OK | Included in output |
| WARN | Included by default; excluded with `--strict` |
| REVIEW | Held for manual review |
| ERROR / FAIL | Excluded from output |

### Exit Codes

| Code | Meaning |
|---:|---|
| 0 | Successful execution |
| 1 | Execution error |
| 2 | One or more processing failures |

## Testing

Run the test suite with:

```bash
python -m unittest tests.test_otoium -v
```

## GUI

Launch the GUI by running the application without arguments:

```bash
python run_otoium.py
```

The GUI provides three tabs:

- **Settings** – Configure processing options.
- **Results** – Inspect generated oto configurations.
- **Logs** – View processing logs.

### Waveform Visualization

The Results tab displays:
- Audio waveforms
- Phoneme boundaries
- Mora processing status

Selecting an alias displays the following oto parameters directly on the waveform:

- Offset
- Overlap
- Preutterance
- Consonant
- Cutoff

Additional interactions:
- Mouse wheel to zoom in and out.
- Drag to navigate the waveform.

The GUI also supports saving and loading settings, as well as cancelling ongoing processing.
