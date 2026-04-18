# Burst Image Restoration and Enhancement

Phase 1: https://drive.google.com/file/d/1XVCyhacwbnuwjHHYuI2ZXHc55WkGemkg/view?usp=drive_link

Phase 2: https://drive.google.com/file/d/1n3CaZUHZ25iU-m8BBelM2nL8_HcyQLgg/view?usp=drive_link

> **Abstract:** *Modern handheld devices can acquire burst image sequence in a quick succession. However, the individual acquired frames suffer from multiple degradations and are misaligned due to camera shake and object motions. The goal of Burst Image Restoration is to effectively combine
complimentary cues across multiple burst frames to generate high-quality outputs. Towards this goal, we develop a novel approach by solely focusing on the effective information exchange between burst frames, such that the degradations get filtered out while the actual scene details are preserved and enhanced. Our central idea is to create a set of pseudo-burst features that combine complimentary information from all the input burst frames to
seamlessly exchange information. However, the pseudo burst cannot be successfully created unless the individual burst frames are properly aligned to discount interframe movements. Therefore, our approach initially extracts pre-processed features from each burst frame and matches them using an edge-boosting burst alignment module. The pseudo-burst features are then created and enriched using multi-scale contextual information. Our final step is to adaptively aggregate information from the pseudo-burst features to progressively increase resolution in multiple stages while merging the pseudo-burst features. In comparison to existing works that usually follow a late fusion scheme with single-stage upsampling, our approach performs favorably, delivering state-of-the-art performance on burst super-resolution, burst low-light image enhancement and burst denoising tasks.* 
<hr />

## Network Architecture

<img src = 'block_diagram.png'> 

## Installation

See [install.yml](install.yml) for the installation of dependencies required to run Restormer.
```
conda env create -f install.yml
```
