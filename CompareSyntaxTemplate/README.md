# CompareSyntaxTemplate Folder

This folder contains scripts, data, and results for comparing syntax patterns between GPT-generated paraphrases and the crowdsourced paraphrases provided by [Jorge et al.](https://link.springer.com/chapter/10.1007/978-3-031-07472-1_15).

### Purpose of the Folder
The `CompareSyntaxTemplate` folder provides tools and datasets for analyzing syntax patterns across GPT-generated and crowdsourced paraphrases. For more details, refer to the provided notebooks and data files.

Below is a description of each file and its purpose:

## Files and Their Descriptions

### 1. `count_syntax.ipynb`
A Jupyter Notebook containing scripts to analyze and count syntax templates that are:
- Present in GPT-generated datasets (`GPT-T/P-12`) but absent in crowdsourced datasets (`Crowd-T/P`).
- Present in crowdsourced datasets (`Crowd-T/P`) but absent in GPT-generated datasets (`GPT-T/P-12`).

The goal is to identify distinctive syntactic structures in GPT-generated versus human-generated paraphrases.

---

### 2. `Crowd_Paraphrases_extraction.ipynb`
A Jupyter Notebook to extract, process and evaluate paraphrases provided by [Jorge et al.](https://link.springer.com/chapter/10.1007/978-3-031-07472-1_15). The description of this notebook and its purpose is provided within the notebook itself.

---

### 3. `Jorge-main-all-with-bertscores.csv`
A CSV file containing all paraphrases from the crowdsourcing study by Jorge et al. It includes BERT scores measuring the similarity between the generated paraphrases and the seed utterances.

---

### 4. `Jorge_bootstrap_correct_only.csv`
A CSV file used in the analysis that includes only the valid paraphrases from the Bootstrap round by [Jorge et al.](https://link.springer.com/chapter/10.1007/978-3-031-07472-1_15) This dataset contains paraphrases that are both semantically related (BERTscore ≥ 0.5) and non-duplicate (BERTscore < 0.98), ensuring high-quality and meaningful paraphrase selection..

---

### 5. `bootstrap-clean.csv`
Another CSV file used in the analysis, containing cleaned data from GPT-bootstrap round.

---

### 6. `pattern_by_example_12-raw.csv`
A dataset of paraphrases generated by the "Pattern_by_example" promptwith 12 paraphrases per request configurations as described in the paper.

---

### 7. `taboo_patterns_12-raw.csv`
A dataset of paraphrases generated by the "Taboo Patterns" prompt with 12 paraphrases per request configurations as described in the paper.
