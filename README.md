# TADI: Traitement Avancé des Images (Advanced Image Processing)
This repository contains practical works (PW) I completed (other students) as part of my master's advanced image processing course (https://perso.telecom-paristech.fr/bloch/P6Image/TADI.html).

## Table of Contents

- [PW n°1: Mathematical Morphology](#pw-n1-mathematical-morphology)
- [PW n°2: Deformable Models](#pw-n2-deformable-models)
- [PW n°3: Markov Models](#pw-n3-markov-models)
- [PW n°4: Markov Random Fields Models](#pw-n4-markov-random-fields-models)
- [PW n°5: Graphcut-Based Approaches](#pw-n5-graphcut-based-approaches)
- [PW n°6: Scale Space](#pw-n6-scale-space)

## PW n°1: Mathematical Morphology
This section contains Python code (Jupyter notebook) and instructions for processing images with morphological operators, and a report presenting the results.
- **Instructions**: [PDF](https://github.com/pictoune/TADI/blob/main/PW_math_morpho/instructions_PW_math_morpho.pdf)
- **Code**: [PW_morpho.ipynb](https://github.com/pictoune/TADI/blob/main/PW_math_morpho/PW_morpho.ipynb)
- **Report**: [PDF](https://github.com/pictoune/TADI/blob/main/PW_math_morpho/report_PW_math_morpho.pdf)

## PW n°2: Deformable Models
This section contains Python (Jupyter notebook) code and instructions for segmenting images with deformable models, and a report presenting the results.
- **Instructions**: [PDF](https://github.com/pictoune/TADI/blob/main/PW_deformable_models/instructions_PW_deformable_models.pdf)
- **Code**: [deformable_models.ipynb](https://github.com/pictoune/TADI/blob/main/PW_deformable_models/deformable_models.ipynb)
- **Report**: [PDF](https://github.com/pictoune/TADI/blob/main/PW_deformable_models/rapport_PW_modeles_deformables.pdf)

## PW n°3: Markov Models
This section contains Python code (Jupyter notebook) and instructions for processing images with Markov models, and a report presenting the results.
- **Code, Instructions & Report**: [PW_Markov.ipynb](https://github.com/pictoune/TADI/blob/main/PW_Markov.ipynb)

## PW n°4: Markov Random Fields Models
This section contains Python code (Jupyter notebook) and instructions for processing images with Markov Random Fields models, and a report presenting the results.
- **Code, Instructions & Report**: [PW_MRF.ipynb](https://github.com/pictoune/TADI/blob/main/PW_MRF.ipynb)

## PW n°5: Graphcut-Based Approaches
This section contains Python code (Jupyter notebook) and instructions for processing images with graph-cut-based approaches, and a report presenting the results.
- **Code, Instructions & Report**: [PW_graphcut.ipynb](https://github.com/pictoune/TADI/blob/main/PW_graphcut/PW_graphcut_part_1.ipynb)

## PW n°6: Scale Space
This section contains Python code (Jupyter notebook) for computing the optical flow of images, and a report presenting the results.
- **Code**: [code](https://github.com/pictoune/TADI/tree/main/PW_scale_space/code)
- **Report**: [PDF](https://github.com/pictoune/TADI/blob/main/PW_scale_space/report_scale_space.pdf)

## Usage 
### Step 1: Clone the Repository
Clone the TADI repository to your local machine using the following commands:
```bash
git clone https://github.com/pictoune/TADI.git
cd TADI
```
### Step 2: Create the required Conda Environment
Set up the required environment using Conda:
  ```bash
  conda env create -f environment.yml -n TADI_env
  ```
### Step 3: Running the code
- If your practical work (PW) code is written in *.py* files, you must first activate the conda environment: 
  ```bash
  conda activate TADI_env
  ```
  then you can run it:
  ```bash
  python <script_name>.py
  ```
- Otherwise if it is written in a *jupyter notebook*, you need to initiate the notebook first:
  ```bash
    jupyter-notebook
  ```
  Once Jupyter Notebook is open, navigate to the notebook you want to run. Then, change the kernel to the TADI environment:
  Go to `Kernel` -> `Change kernel` -> `Python [conda env:TADI_env]`.
  Then you can run the cells.

## License

This project is open source and available under the [MIT License](LICENSE).

Feel free to explore the projects and reach out if you have any questions or suggestions.
