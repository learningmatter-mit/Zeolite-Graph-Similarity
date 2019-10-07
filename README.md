# Graph Similarity Drives Zeolite Diffusionless Transformations and Intergrowth

Compares zeolite structures using graphs and structural similarities. This implementation in Python was made by Daniel Schwalbe-Koda. If you use this code, please cite 

D. Schwalbe-Koda, Z. Jensen, E. Olivetti and R. Gómez-Bombarelli. "Graph similarity drives zeolite diffusionless transformations and intergrowth". _Nature Materials_ (2019). Link: https://www.nature.com/articles/s41563-019-0486-1

BibTeX format:

```
@article{10.1038/s41563-019-0486-1,
    title={Graph similarity drives zeolite diffusionless transformations and intergrowth},
    author={Schwalbe-Koda, Daniel and Jensen, Zach and Olivetti, Elsa and Gómez-Bombarelli, Rafael},
    journal={Nature Materials},
    year={2019},
    doi={10.1038/s41563-019-0486-1}
}
```

## Contents

This repository contains:

* [Data](data/) of the Supplementary Information of the article.
* [Code](zeograph/) implementing the equations of relevance of the article.
* [Tutorials](tutorials/) on how to use our code and reproduce our results. We cover:
    * How to [open and read the data](tutorials/data.ipynb) of the article
    * How to [verify the existence of isomorphism](tutorials/isomorphism.ipynb) between periodic graphs of zeolite structures
    * How to [calculate the D-measure](tutorials/dmeasure.ipynb) between graphs
    * How to [calculate the SOAP distance](tutorials/soap.ipynb) between zeolites
    * How to [match supercells and calculate the graph distance](tutorials/supercells.ipynb) using our methods of supercell matching.

## Usage

The tutorials are self-explanatory. This code was tested with the following dependencies:

```
python==3.7.4,
numpy==1.16.4,
ase==3.18.1
pymatgen==2019.5.8,
soaplite==1.0.3,
networkx==2.3
```

To use the code, you can create an [anaconda](https://conda.io/docs/index.html) environment. Learn more on how to manage anaconda environments by reading [this page](http://conda.pydata.org/). To create a new environment working with the current code, run the following commands:

```bash
conda create -n zeograph python=3.7 numpy networkx pymatgen=2019.5.8 ase -c anaconda -c matsci -c conda-forge
```

After the installation is complete, install `soaplite` by entering the environment and installing this package using `pip`:

```bash
conda activate zeograph
pip install soaplite
```

This should provide an environment adequate for running the tutorials.
