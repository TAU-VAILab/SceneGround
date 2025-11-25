# Scene Grounding In the Wild

This is the official implementation of Scene Grounding In The Wild.

[![arXiv]()]()

[[Project Website]()]

> **Scene Grounding In the Wild**<br>
> Tamir Cohen<sup>1</sup>, Leo Segre <sup>1</sup>, Shay Shomer Chai<sup>1</sup>, Shai Avidan <sup>1</sup>, Hadar Averbuch-Elor <sup>2</sup><br>
> <sup>1</sup>Tel Aviv University, <sup>2</sup>Cornell University

>**Abstract** <br>
>               Reconstructing accurate 3D models of large-scale real-world scenes from unstructured, in-the-wild imagery remains a core challenge in computer vision, especially when the input views have little or no overlap. In such cases, existing reconstruction pipelines often produce multiple disconnected partial reconstructions or erroneously merge non-overlapping regions into overlapping geometry. 
In this work, we propose a framework that grounds each partial reconstruction to a complete reference model of the scene, enabling globally consistent alignment even in the absence of visual overlap. 
We obtain reference models from dense, geospatially accurate pseudo-synthetic renderings derived from Google Earth Studio. These renderings provide full scene coverage but differ substantially in appearance from real-world photographs. Our key insight is that, despite this significant domain gap, both domains share the same underlying scene semantics. We represent the reference model using 3D Gaussian Splatting, augmenting each Gaussian with semantic features, and formulate alignment as an inverse feature-based optimization scheme that estimates a global 6DoF pose and scale while keeping the reference model fixed. 
Furthermore, we introduce the WikiEarth dataset, which registers existing partial 3D reconstructions with pseudo-synthetic reference models. We demonstrate that our approach consistently improves global alignment when initialized with various classical and learning-based pipelines, while mitigating failure modes of state-of-the-art end-to-end models.

![Teaser](./images/teaser.png)
</br>

## BibTeX
If you find our work useful in your research, please consider citing:
    
</br>

