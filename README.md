# MolImgDiff

Implementation of "Diffusion Model for Molecular Generation: An Image Perspective"

![](main.jpeg?v=1&type=image)

## Dependency:

All the dependent packages and versions of our runtime environment can be found in the `requirement.txt` file

## Data preparation:

1.  Prepare molecular images

    `cd preparaton`

    `python 01_smi2img.py`

2.  Prepare ESM2 model

    Download parameters of the ESM2 model and tokeniser from [here](https://github.com/facebookresearch/esm 'here'). Then place the downloaded file under `preparation/esm/`

3.  Prepare PDB embedding

    `cd preparaton`

    `python 02_get_emb.py`

## Train:

If you want to train on a single GPU, run the following command:

    python mol_train.py \
      --data_dir datasets/chembel_512/chembel_pic \
      --diffusion_step 1000 \
      --lr_anneal_steps 160000

If you want to train on multiple GPUs, run the following command:

    mpiexec -n 2 python mol_train.py \
      --data_dir datasets/chembel_512/chembel_pic \
      --diffusion_step 1000 \
      --lr_anneal_steps 160000

## Test:

To replicate our results, download the pre-trained checkpoints from [here](https://drive.google.com/file/d/1FM-QtH2Bqy2VKT9eLgbgPvzsjWOWdmmR/view?usp=sharing).&#x20;

    mpiexec -n 2 python mol_sample.py
      --model_path ../logger_chembel_line_target_2k_16w_bt10/model160000.pt
      --diffusion_step 1000
      --num_samples 1100
      --use_pdb True
      --batch_size = 80

Note: Our code was developed with reference to the code written by [Nichol et al.](https://proceedings.mlr.press/v139/nichol21a.html), and we would like to express our gratitude to them.&#x20;

If you have any questions, feel free to contact Hao Lu, <luhao@stu.ouc.edu.cn>
