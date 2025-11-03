# Corpus Phrase Miner: Discovering Meaningful Phrases Through Statistical Analysis

This toolkit helps researchers and analysts discover meaningful phrases in specialized text corpora—whether from conversations, interviews, documents, or any domain-specific collection. It uses computational linguistics methods to identify word sequences (**n-grams**) that behave as cohesive units of meaning, revealing patterns that may not be visible through simple keyword analysis.

The approach goes beyond simple frequency counting by identifying phrases that are:
- **Cohesive**: Words that appear together more than chance would predict
- **Domain-specific**: Phrases characteristic of your corpus vs general language
- **Adopted**: Used by multiple speakers/sources, indicating shared concepts
- **Meaningful**: Statistically significant and contextually relevant

Starting with thousands of possible word combinations, we systematically filter them using multiple statistical measures to surface the phrases most likely to represent important concepts, technical terms, or emergent themes in your data.

## Our Strategy

### Extract phrases (n-grams) from your corpus
- From your text data, we extract every possible bigram and trigram.
- (Imagine a sentence with each possible pair/triple of words underlined — that's our raw search space.)
        
### Filter for meaningful combinations
- We first count frequency.
- To distinguish mundane word pairings from real phrases, we use **mutual information (MI)**.
- High MI means the words appear together much more often than chance, suggesting a meaningful unit.
        
### Check adoption across speakers
- A phrase used only once or by a single person is unlikely to represent a shared frame.
- We therefore count how many different speakers used each phrase.
    
### Separate the novel from the mundane by comparing with reference corpora (domain specificity)
    
- Many high-MI phrases are ordinary, like "I know."
- We look for phrases not found in general English corpora to identify domain-specific terminology.
- We also flag phrases that exist in everyday English but occur significantly more often in your corpus — indicating **domain specificity**.
        
### Compare MI across corpora
    
- Some phrases are domain-specific but may not be conceptually meaningful.
- By comparing MI between your corpus and a general corpus, we ask: _Is this phrase especially cohesive or characteristic in your specific context?_
- A positive difference suggests distinctive usage patterns; negative or neutral differences suggest generic language.
        
### Visualise and contextualise usage

- Identified candidate phrases can be plotted on a timeline to understand patterns and illustrate how candidate phrases function in conversation.
    - Who used it first?
    - Who picked it up later?
    - How frequently did it recur?

### Cluster similar phrases to see evolving themes
- By grouping semantically similar phrases, you can track how concepts and themes evolve over time or across different sources.



## What Each Linguistic Measure Means 

- **Frequency (corpus/reference)**: how often a phrase is used in each corpus (size-normalized). Separates one-offs from phrases that gain traction.
- **Speaker/Source coverage**: how many different speakers or sources used the phrase. Shared phrases are more likely to reflect collective concepts than individual idiolects.
- **Mutual Information (MI)**: how strongly the words in a phrase are "tied together" (above chance). High-MI items behave like meaningful units (e.g., _machine learning_, _climate change_), not random juxtapositions.
- **Domain specificity (log2 ratio + p-value)**: how much more common a phrase is in your corpus than in general language, with statistical confidence. This surfaces domain-specific terminology while filtering out ubiquitous phrases like _I think_.
- **MI difference (MI z diff)**: how much more **cohesive** a phrase is in your corpus than in general language. Distinguishes domain terms that are merely topical from phrases whose **association** is especially tight in your specific context.

## Overview of the Notebook Series

- **01 - finding cohesive n-grams (mutual information score).ipynb — Build phrase inventory & association strength**  
  - Extract unigrams/bigrams/trigrams, quantify frequencies, inspect distributions
  - Compute **Mutual Information (MI)** scores
  - Normalize by n-gram length for fair comparison
  - Track speaker/source coverage
  - Save results for further analysis
    
- **02 - finding domain-specific n-grams (relative frequency).ipynb — Compare frequencies with reference corpora**  
  - Use general language corpora (e.g., Brown, Switchboard) as baseline
  - Compute **log frequency ratios** and statistical significance
  - Identify phrases unique to your corpus and those statistically overrepresented
  - Export comparison results
    
- **03 - finding characteristic n-grams (relative cohesion).ipynb — Find "characteristic" phrases**  
  - Compare MI scores between your corpus and reference
  - Calculate normalized MI differences
  - Identify phrases with distinctive cohesion patterns
  - Save characteristic phrase rankings
    
- **04 - visually explore usage patterns.ipynb — Explore patterns and context**  
  - Visualize phrase distributions over time or across sources
  - Generate interactive plots for exploration
  - Extract examples in context (KWIC)
  - Optionally cluster semantically similar phrases to identify themes

## Environment and Python version

This project is tested with **Python 3.9.6**. Some dependencies (e.g., VegaFusion/Altair integrations) work best on this version.

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate vis-analysis
python -m ipykernel install --user --name vis-analysis --display-name "Python 3.9.6 (vis-analysis)"
```

### Option B: pyenv + venv

```bash
# install pyenv (macOS)
brew install pyenv
pyenv install 3.9.6
pyenv local 3.9.6   # repo includes .python-version for auto-switching

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name vis-analysis --display-name "Python 3.9.6 (vis-analysis)"
```

Notes:
- The repository includes a `.python-version` file so `pyenv` automatically selects 3.9.6 when you `cd` into the project.
- If installing packages from a system-managed Python shows a PEP 668 error, use a virtual environment (Conda or `python -m venv`) and install inside that environment.