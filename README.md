## AVSBench-Robust

This repository provides the official dataset generation, training, and evaluation scripts for:

**Do Audio-Visual Segmentation Models Truly Segment Sounding Objects?**
[[arXiv Link](https://arxiv.org/abs/2502.00358)]

### Overview
Audio-Visual Segmentation (AVS) aims to segment sounding objects in visual scenes. However, many models exhibit a "visual bias", segmenting visually salient objects regardless of whether they are actually producing sound.
AVSBench-Robust is designed to rigorously evaluate the robustness of AVS models against this bias. It extends the original AVSBench dataset by incorporating challenging **negative audio scenarios** for each video, including:
* Silence 
* Noise
* Off-screen Sounds
  
This allows for a comprehensive assessment of whether models truly integrate audio-visual cues or rely solely on visual information.

### Repository Contents

This repository provides all necessary resources for:
* Reconstructing the AVSBench-Robust dataset
* Training AVS models on robust datasets
* Evaluating model robustness using newly designed metrics

### Dataset Generation
Requirements Before Running:
- The original AVSBench datasets ([download instructions here](https://github.com/OpenNLPLab/AVSBench))
- `metadata.csv` from the AVSS dataset (also available [here](https://github.com/OpenNLPLab/AVSBench))

Specifically, the scripts generate new audio versions for each video to simulate negative audio conditions, creating both **misaligned** and **noise** audio.
The main script for generating the robust audio data is:
```bash
python audio_processing.py --dataset [s4|ms3] --seed [random_seed]
```
Arguments
- `--dataset`: Select the target dataset (s4 or ms3).
- `--seed`: Random seed for reproducibility.

### Training
_Coming soon._
We will release training scripts used to reproduce the results in the paper, including:
- Training with robustness augmentation
- Details on optimizer, learning rate, batch size, and hardware used

  
### Evaluation
_Coming soon._
We will provide evaluation scripts to measure:
- Robustness under Negative Audio (False Positive Rate, G-mIoU and G-F)


### Citation
If you find AVSBench-Robust useful for your research, please cite:
```bibtex
@article{li2025audio,
  title={Do Audio-Visual Segmentation Models Truly Segment Sounding Objects?},
  author={Li, Jia and Zhao, Wenjie and Huang, Ziru and Guo, Yunhui and Tian, Yapeng},
  journal={arXiv preprint arXiv:2502.00358},
  year={2025}
}
```
