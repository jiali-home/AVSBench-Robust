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

### 🛠️ Get Started

#### 1. Environments
```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source miniconda3/bin/activate
conda init --all

conda create -n segformer python=3.10 pip
conda activate segformer
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge gcc=12.4.0 gxx=12.4.0
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install pandas, timm, resampy, soundfile, gdown, nvitop, pydub, librosa
pip install "numpy<2"

# build MSDeformAttention

git clone https://github.com/LONZARK/AVSegformer.git
cd AVSegformer/ops
sh make.sh
```


### 2. Dataset preparation

#### AVSBench dataset

Please refer to the link [AVSBenchmark](https://github.com/OpenNLPLab/AVSBench) to download the datasets. You can put the data under `data` folder or rename your own folder. Remember to modify the path in config files. The `data` directory is as bellow:
```
|--data
   |--AVSS
   |--Multi-sources
   |--Single-source
```

#### AVSBench-Robust dataset: 

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
- `--seed`: Random seed for reproducibility. We use 42 in our paper.


### 3. pretrained backbones

The pretrained ResNet50/PVT-v2-b5 (vision) and VGGish (audio) backbones can be downloaded from [here](https://drive.google.com/drive/folders/1386rcFHJ1QEQQMF6bV1rXJTzy8v26RTV?usp=sharing) and placed to the directory `pretrained_backbones`.


### 4.Train
Train on [AVSegformer](https://github.com/vvvb-github/AVSegFormer/blob/master/README.md?plain=1) Baseline
```shell
cd AVSegformer
bash train_s4.sh
bash train_ms3.sh
```

###  5. Test
Test on [AVSegformer](https://github.com/vvvb-github/AVSegFormer/blob/master/README.md?plain=1) Baseline

```shell
cd AVSegformer
bash test_s4.sh 
bash test_ms3.sh 
```

The **False Positive Rate (FPR)** metric is calculated directly in the evaluation code.  
You can find the implementation here:  
👉 [AVSegFormer/scripts/s4/test.py, Line 15](https://github.com/jiali-home/AVSBench-Robust/blob/67c0f2e750cc268bb0e85e666d94b2b28e5fb7bd/training_evaluation/code/AVSegFormer/scripts/s4/test.py#L15)

The G-mIoU, G-F and G-FPR metrics are manually aggregated and calculated based on the exported results, using Excel. 

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
