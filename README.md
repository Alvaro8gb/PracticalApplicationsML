
# Machine Learning Applications


This repository contains practical applications and clustering analyses utilizing machine learning algorithms. The dataset chosen is the following: [Dataset LInk](http://archive.ics.uci.edu/dataset/878/cirrhosis+patient+survival+prediction+dataset-1)
The project structure is organized as follows:

## Practical Application 1 (Non-probabilistic Supervised Classification):

Non-probabilistic classification algorithms, covered in class, are applied in four analyses:

1. With all original variables.
2. With a univariate filter feature subset selection.
3. With a multivariate filter feature subset selection.
4. With a wrapper feature subset selection.

All merit figures are estimated using an honest method.

## Practical Application 2 (Probabilistic Supervised Classification):

Probabilistic classification algorithms, covered in class, are applied similarly to Practical Application 1. Additionally, metaclassifiers are introduced. All merit figures are estimated with an honest method.

## Practical Application 3 (Unsupervised Classification):

Unsupervised classification algorithms covered in class are applied. The dataset does not contain a class variable.

The visualization of each clustering is save in [ref](./clustering/img) and have this structure:
```bash
img
│   │   ├── all_faces.png
│   │   ├── complete
│   │   │   ├── complet_de.png
│   │   │   ├── complet_pca.png
│   │   │   └── complet_tree.png
│   │   ├── partional
│   │   │   ├── partional_pca.png
│   │   │   └── partional_tree.png
│   │   ├── probabilistic
│   │   │   ├── prob_pca.png
│   │   │   └── prob_tree.png
│   │   ├── sample_faces.png
│   │   ├── single
│   │   │   ├── single_de.png
│   │   │   ├── single_PCA.png
│   │   │   └── single_tree.png
│   │   └── ward
│   │       ├── ward_de.png
│   │       ├── ward_pca.png
│   │       └── ward_tree.png

```

## Reports 

The reports of each practical aplication is located in [ref](./reports/)
