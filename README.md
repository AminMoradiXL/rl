# Reinforcement Learning in Physics: A Comprehensive Review

This repository contains the codes of the paper **"Reinforcement Learning in Physics"**, a review paper that surveys various applications of reinforcement learning (RL) across the physical sciences. The paper explores the use of RL in domains such as quantum mechanics, medical physics, neuclear physics, and more, providing a detailed analysis of recent advancements and methodologies.

## Overview

Reinforcement learning has emerged as a powerful tool for solving complex problems in physics, from controlling quantum systems to optimizing materials properties. This paper aims to:

- Provide a thorough review of the state-of-the-art RL applications in physical sciences.
- Categorize the different RL methods used in physics.
- Highlight the challenges and future directions for RL in this domain.

## Contents

- **`/paper/`**: Contains the LaTeX source code, figures, and bibliography for the paper.
- **`/datasets/`**: Includes datasets referenced in the paper, along with preprocessing scripts.
- **`/experiments/`**: Jupyter notebooks and Python scripts used for reproducing results mentioned in the paper.
- **`/figures/`**: High-resolution images of the figures used in the paper.
- **`/code/`**: Contains any code implementations of RL algorithms used or referenced in the paper.
- **`/results/`**: Experimental results, plots, and analysis data.

## Getting Started

### Prerequisites

To compile the paper and run the code, you will need:

- **LaTeX**: For compiling the paper.
- **Python 3.x**: For running the experiments.
- **Jupyter Notebook**: To view and run the analysis.
- **Requirements**: Install dependencies using:
    ```bash
    pip install -r requirements.txt
    ```

### Building the Paper

To compile the LaTeX source code into a PDF, navigate to the `paper/` directory and run:

```bash
pdflatex main.tex
```

## Running Experiments
You can run the experiments using the Jupyter notebooks provided in the ```experiments/``` folder:

```bash
cd experiments
jupyter notebook
```

Open the desired notebook and run the cells to reproduce the results.

## Contributing
We welcome contributions to this repository, including improvements to the paper, code, or documentation. To contribute:

Fork the repository.
Create a new branch (```git checkout -b feature/YourFeature```).
Commit your changes (```git commit -am 'Add new feature'```).
Push to the branch (```git push origin feature/YourFeature```).
Open a pull request.
Please ensure that your contributions adhere to the repository's coding standards and include tests where applicable.

## Citation
If you use any part of this work in your own research, please consider citing:

```css
@article{moradi2024RLinPhysics,
  author = {Mohammadamin Moradi, Lili Ye, Ying-Cheng Lai},
  title = {Reinforcement Learning in Physics: A Comprehensive Review},
  journal = {TBA},
  year = {2024},
  volume = {XX},
  number = {YY},
  pages = {ZZZ-AAA},
  doi = {10.XXXX/yourdoi},
}
```

## License
This repository is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any questions or inquiries, please contact Mohammadamin Moradi - mmoradi5@asu.edu.
