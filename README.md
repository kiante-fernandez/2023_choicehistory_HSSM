# History-Dependent Decision-Making across Species

This repository hosts the code and supplementary materials for the paper "History-Dependent Decision-Making across Species." Our research delves into the decision-making behaviors of humans and mice, revealing a shared cognitive strategy of history-dependent evidence accumulation.

> **History-Dependent Decision-Making across Species**  
> Kianté Fernnadez<sup>2</sup>, Alexander Fengler<sup>2</sup>, and Anne Urai<sup>3</sup>  
> <sup>1</sup>Department of Psychology, University of California, Los Angeles, CA, USA  
> <sup>2</sup>Department of Cognitive, Linguistic and Psychological Sciences, Brown University, USA  
> <sup>3</sup>Department of Psychology, Leiden University, The Netherlands 

## Abstract
In our study, we analyze decision-making behaviors in humans and mice, uncovering a common cognitive strategy involving history-dependent evidence accumulation. The study reveals that individual differences in choice repetition are influenced by a history-dependent bias in the rate of evidence accumulation, rather than its starting point, emphasizing the importance of evidence integration across multiple temporal scales. This discovery is crucial for understanding decision-making processes across mammalian species and sets the foundation for future research into the neural dynamics of these phenomena.

## Requirements

- [HSSM - Hierarchical Sequential Sampling Modeling](https://github.com/lnccbrown/HSSM)
- [Package to train likelihood-approximation networks](https://github.com/AlexanderFengler/LANfactory)
- [iblenv](https://github.com/int-brain-lab/iblenv)

## Repository Contents
- `data/` - Datasets used in the study.
- `src/` - Source code for models and analysis.
- `docs/` - Additional documentation and supplementary materials.
- `results/` - Results and visualizations from the analysis.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/kiantefernandez/2023_choicehistory_HSSM.git
   cd 2023_choicehistory_HSSM
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   conda activate iblenv
   ```

## Usage

1. After installation, access the source code directory:
   ```bash
   cd src
   ```

2. Execute the analysis scripts in Python:
   ```python
   get_data.py # Retrieves data from IBL public server
   preprocess_data.py # Selects suitable RTs
   figure1a_plot_behavior.py # Plots basic data insights
   figure1b_choice_history.py # Fits psychometric functions with history terms
   figure1c_history_strategy.py
   ```

### Contribute to the repo

We advocate for a lightweight, but still somewhat rigorous approach to collaborative coding in this repo.
Here are some basic principles and minimal examples for corresponding terminal commands:

1. Never work on the main branch directly. Instead begin your work in one of two ways:

 This avoids unnecessary conflicts that stall work leading to tedious git debugging, and moreover keeps a cleaner record of tasks completed via merges to main (see below). 
 In your local terminal type `git checkout -b my-expressive-branch-name`. This will switch you to the branch `my-expressive-branch-name`. You complete your current task in there, then push. Follow the instructions in the terminal when pushing from your local branch. Git will ask you to execute the following terminal command: 'git push --set-upstream origin my-expressive-branch-name'. We can then turn this branch into a pull-request (PR) (this is an option on the github page gui as well). Pull-requests can then be reviewed, commented, revised and eventually merged into the main branch.
 Once a branch is merged into main, it can be deleted.

 If you are currently unsure which branch your are on, type `git branch -a` which will list all the branches available and highlight the one your are currently working in.

 You can in general switch between pre-existing branches via `git checkout branch-name`.

1. For every commit, do your best to capture what was done in a concise commit message. Pushing a commit will usually follow the following sequence:

- `git add .` (add new files to be tracked, you may sometimes not want this)
- `git commit -m 'my expressive commit message'`
- `git push` (this pushes the changes to github, if you do this from a new branch that you defined locally, you have to follow the instruction on the screen here)

3. Comment your functions, including all input arguments. At best, directly use valid doc strings. [Here](https://www.datacamp.com/tutorial/docstrings-python) is a little tutorial on doc strings in case you wonder about the right format. This is important for collaborators / reviewers of pull requests to digest the new code.

## Citing This Work
If you use this research or the accompanying code in your work, please cite:
```
Urai, A., & Brain Laboratory, T. I. (2023). History-Dependent Decision-Making across Species. 2023 Conference on Cognitive Computational Neuroscience. Oxford, UK. https://doi.org/10.32470/CCN.2023.1119-0
```

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/kiante-fernandez/2023_choicehistory_HSSM/blob/main/LICENSE) file for details.

## Contact
For queries, please reach out to the corresponding author Anne Urai, Leiden University, at a.e.urai@fsw.leidenuniv.nl.

