# Citation

This code is a modified fork of MatterGen. If you use it, please also cite the upstream MatterGen
paper:

```bibtex
@article{MatterGen2025,
  author  = {Zeni, Claudio and Pinsler, Robert and Z{\"u}gner, Daniel and Fowler, Andrew and Horton, Matthew and Fu, Xiang and Wang, Zilong and Shysheya, Aliaksandra and Crabb{\'e}, Jonathan and Ueda, Shoko and Sordillo, Roberto and Sun, Lixin and Smith, Jake and Nguyen, Bichlien and Schulz, Hannes and Lewis, Sarah and Huang, Chin-Wei and Lu, Ziheng and Zhou, Yichi and Yang, Han and Hao, Hongxia and Li, Jielan and Yang, Chunlei and Li, Wenjie and Tomioka, Ryota and Xie, Tian},
  journal = {Nature},
  title   = {A generative model for inorganic materials design},
  year    = {2025},
  doi     = {10.1038/s41586-025-08628-5},
}
```

## Methods this fork builds on

Atom-type diffusion (D3PM is upstream MatterGen's default; MDLM and Duo are added here, with the
samplers and SUBS parameterisation adapted from the official implementations):

```bibtex
@inproceedings{austin2021d3pm,
  author    = {Austin, Jacob and Johnson, Daniel D. and Ho, Jonathan and Tarlow, Daniel and van den Berg, Rianne},
  title     = {Structured Denoising Diffusion Models in Discrete State-Spaces},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021},
}

@inproceedings{sahoo2024mdlm,
  author    = {Sahoo, Subham Sekhar and Arriola, Marianne and Schiff, Yair and Gokaslan, Aaron and Marroquin, Edgar and Chiu, Justin T. and Rush, Alexander and Kuleshov, Volodymyr},
  title     = {Simple and Effective Masked Diffusion Language Models},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2024},
}

@inproceedings{sahoo2025duo,
  author    = {Sahoo, Subham Sekhar and Deschenaux, Justin and Gokaslan, Aaron and Wang, Guanghan and Chiu, Justin and Kuleshov, Volodymyr},
  title     = {The Diffusion Duality},
  booktitle = {International Conference on Machine Learning},
  year      = {2025},
}
```

Diversity metrics (coverage and Wasserstein distances, with the fingerprints taken from DiffCSP's
evaluation code):

```bibtex
@inproceedings{xie2022cdvae,
  author    = {Xie, Tian and Fu, Xiang and Ganea, Octavian-Eugen and Barzilay, Regina and Jaakkola, Tommi},
  title     = {Crystal Diffusion Variational Autoencoder for Periodic Material Generation},
  booktitle = {International Conference on Learning Representations},
  year      = {2022},
}

@inproceedings{jiao2023diffcsp,
  author    = {Jiao, Rui and Huang, Wenbing and Lin, Peijia and Han, Jiaqi and Chen, Pin and Lu, Yutong and Liu, Yang},
  title     = {Crystal Structure Prediction by Joint Equivariant Diffusion},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2023},
}
```

Oxidation-state metrics (charge-neutrality and Pauling screening, and ICSD species-occurrence
statistics, from SMACT):

```bibtex
@article{davies2019smact,
  author  = {Davies, Daniel W. and Butler, Keith T. and Jackson, Adam J. and Skelton, Jonathan M. and Morita, Kazuki and Walsh, Aron},
  title   = {{SMACT}: Semiconducting Materials by Analogy and Chemical Theory},
  journal = {Journal of Open Source Software},
  volume  = {4},
  number  = {38},
  pages   = {1361},
  year    = {2019},
  doi     = {10.21105/joss.01361},
}
```
