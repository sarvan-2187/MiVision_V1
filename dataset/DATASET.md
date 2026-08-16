# Dataset

This project does not include the raw dataset in the repository (large 
binary image data doesn't belong in git history, and the original license 
is unspecified - see [Licensing](#licensing) below). This document explains 
where the data came from, how it was obtained, and the reasoning behind 
which classes were selected for training.

---

## Source

**[Military Aircraft Detection Dataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)** 
- Kaggle, uploaded by `a2015003713`, 298 upvotes at time of use.

- **Scope:** 114 total classes - 103 military aircraft types + 11 commercial 
  airliner classes (added by the original author to broaden detection 
  difficulty)
- **Size:** ~92,100 files, 12.81 GB total
- **Format:** JPEG images with a single `labels_with_split.csv` at the 
  dataset root containing bounding box annotations in pixel space 
  (`filename, width, height, class, xmin, ymin, xmax, ymax, split`)
- **Annotations:** 43,418 labeled instances across all 114 classes; 
  pre-defined `train` / `validation` / `test` split included in the CSV
- **Also included:** a `crop/` directory with pre-cropped, per-class single-
  aircraft images (useful for a classification approach, not used directly 
  in this detection pipeline), and an `annotated_samples/` directory of 
  sample visualizations

### Why this dataset over alternatives

Before settling on this dataset, several alternatives were evaluated:

| Dataset | Why not used |
|---|---|
| AeroScan (Kaggle, 88 classes, Apache 2.0) | No `data.yaml` or class-mapping file anywhere in the download; only 11 upvotes and zero community validation (0 code, 0 discussion); cover sample image showed visible label-quality issues (duplicate/ambiguous class tags on the same aircraft) |
| FGVC-Aircraft (academic benchmark) | Civil aircraft variant classification (e.g., 737-700 vs 737-800), not military; no bounding boxes - classification only, not detection |
| HRPlanes (peer reviewed, DOI, YOLO ready) | Single-class only ("airplane") - aerial/satellite detection, no aircraft-type classification, so it doesn't fit a "classify which military aircraft" goal |

This dataset was chosen because it had the strongest combination of: real 
community validation (298 upvotes, active discussion), bounding box 
annotations in a parseable (if non-standard) format, and actual military 
aircraft type labels rather than single-class or civil-only coverage.

---

## How to obtain it

The dataset was downloaded programmatically via `kagglehub` rather than the 
Kaggle CLI (the CLI's `-f` selective-file download returned repeated 404s 
against this specific dataset; `kagglehub.dataset_download()` worked 
reliably, at the cost of pulling the full 12.81GB archive rather than a 
single file):

```python
import kagglehub

path = kagglehub.dataset_download("a2015003713/militaryaircraftdetectiondataset")
print("Dataset downloaded to:", path)
```

This requires a Kaggle account and API token (`kaggle.json`) configured in 
your environment. See [Kaggle's API documentation](https://www.kaggle.com/docs/api) 
for setup.

The dataset ships with **no `data.yaml` or class-name mapping file** - 
class identity must be derived directly from the `class` column in 
`labels_with_split.csv`, not from filenames (most filenames are content 
hashes with no embedded class information;

---

## Class selection

The full dataset covers 114 classes with severe long-tail imbalance - the 
largest class (F16) has 2,104 instances, while several rare classes have 
under 20. Training on all 114 classes within this project's scope would 
have meant most classes having too few examples to learn from meaningfully, 
and no realistic way to validate label quality across that many categories.

**17 classes were selected**, filtered from the real per-class instance 
counts in the CSV, using two criteria:

1. **Adequate sample size** - every selected class has 190+ instances, with 
   most in the 500-2,000+ range.
2. **Visual distinctiveness** - classes were deliberately chosen to span 
   different aircraft categories (fighters, stealth aircraft, bombers, 
   large transports, helicopters, VTOL) rather than clustering multiple 
   near-identical silhouettes together (e.g., avoiding grouping several 
   generic delta-wing fighter jets that are hard to visually distinguish 
   even for a human, which the dataset's own confusion patterns later 
   confirmed as a real risk for similar-looking classes that *were* included, 
   like F16/F18).

Commercial airliner classes (Airbus A320/A330/A340/A350/A380, Boeing 
737/747/757/767/777/787) were excluded entirely, since the goal was 
specifically military aircraft detection.

**Final 17 classes and instance counts:**

| Class | Instances | Category |
|---|---|---|
| F16 | 2,104 | Fighter |
| F18 | 1,833 | Fighter |
| C130 | 1,610 | Transport |
| F35 | 1,596 | Stealth fighter |
| F15 | 1,586 | Fighter |
| V22 | 936 | VTOL |
| J20 | 928 | Stealth fighter |
| C17 | 759 | Transport |
| F22 | 726 | Stealth fighter |
| B52 | 641 | Bomber |
| B2 | 581 | Stealth bomber |
| AH64 | 573 | Helicopter |
| F14 | 560 | Fighter |
| Mig31 | 510 | Fighter |
| Su57 | 502 | Stealth fighter |
| CH47 | 397 | Helicopter |
| Tejas | 192 | Fighter |

**A note on Tejas:** it was included deliberately, as the sole 
Indian-designed aircraft in the dataset (HAL Tejas), despite having the 
smallest instance count by a wide margin (~11x fewer than F16). This was a 
conscious tradeoff - it was expected going in that this class would 
underperform relative to the others due to limited training data, and that 
is exactly what happened (see main [README](README.md#results) for the 
resulting metrics). It's kept in and disclosed rather than dropped, because 
the honest failure mode is more informative than a clean 16-class result 
with a class of personal interest silently excluded.

Aircraft considered but **not** included despite Indian military relevance: 
Rafale and JF-17 appear in the dataset but are not Indian-designed 
(French and Chinese/Pakistani respectively, despite Rafale being IAF-
operated); no other Indian programs (e.g., HAL Dhruv, Prachand) exist in 
this dataset's 114-class list.

---

## Format conversion

The source CSV provides bounding boxes as absolute pixel coordinates 
(`xmin, ymin, xmax, ymax`). YOLO requires normalized center-format 
coordinates (`class_id x_center y_center width height`, all values 0-1). 
A custom script (not part of Ultralytics' tooling, since no pre-existing 
YOLO labels shipped with this dataset) performs this conversion:

1. Filters the CSV to the 17 selected classes
2. Builds a class name → integer ID mapping (0-16)
3. Groups all bounding boxes per image filename
4. Converts each box: 
   `x_center = ((xmin + xmax) / 2) / image_width`, 
   `y_center = ((ymin + ymax) / 2) / image_height`, 
   `width = (xmax - xmin) / image_width`, 
   `height = (ymax - ymin) / image_height`
5. Writes one `.txt` label file per image and copies the corresponding 
   source image into a `train/validation/test` folder structure, using the 
   `split` column already present in the source CSV
6. Generates `data.yaml` pointing to the resulting folders

Result: 8,528 images (train: 6,326 / validation: 1,621 / test: 581), zero 
missing image-to-annotation mismatches confirmed during conversion.

---

## Licensing

The original Kaggle dataset lists its license as **unspecified/unknown**. 
This project uses it strictly for **non-commercial, educational, and 
portfolio purposes**. If you intend to use this dataset or derived models 
for anything beyond that, verify image rights independently - an 
unspecified license on Kaggle does not necessarily mean the underlying 
images are freely redistributable, only that the uploader didn't declare 
terms.

---

## Reproducing this filtering step

If you have downloaded the dataset yourself and want to reproduce the same
17-class filtered subset and YOLO conversion, use the scripts in
[scripts/](../scripts/) - see the [Scripts section of the README](../README.md#scripts)
for what each one does and how to run it.