# 3D-MetaFormer

`MetaFormer` families with Temporal `UNETR` Architecture for 3D medical image segmentation

## Dependencies

```requirements.txt
monai[tqdm,nibabel,einops,tensorboardX,matplotlib]==1.3.0
torch==2.1.0
torchvision==0.16.0
timm==0.9.8
```

```bash
pip install -r requirements.txt
```

## Models

Please download the Self-supervised pre-trained weights for `SwinUNETR` backbone from this <a href="">link</a>

<table>
    <tr>
        <th>Name</th>
        <th>Dice (overlap=0.5)</th>
        <th>Feature Size</th>
        <th># Params (M)</th>
        <th>Self-Supervised Pre-trained</th>
        <th>Checkpoint</th>
    </tr>
    <tr>
        <td>SwinUNETR base</td>
        <td>81.86</td>
        <td>48</td>
        <td>62.1</td>
        <td>Yes</td>
        <td><a href="https://github.com/BlueDruddigon/3D-MetaFormer/releases/download/0.1.2/swin_unetr.base_5000ep_lr2e-4.pt">Download</a></td>
    </tr>
    <tr>
        <td>SwinUNETR small</td>
        <td>79.34</td>
        <td>24</td>
        <td>15.7</td>
        <td>No</td>
        <td><a href="https://github.com/BlueDruddigon/3D-MetaFormer/releases/download/0.1.2/swin_unetr.small_5000ep_lr2e-4.pt">Download</a></td>
    </tr>
    <tr>
        <td>SwinUNETR tiny</td>
        <td>70.35</td>
        <td>12</td>
        <td>4.0</td>
        <td>No</td>
        <td><a href="https://github.com/BlueDruddigon/3D-MetaFormer/releases/download/0.1.2/swin_unetr.tiny_5000ep_lr2e-4.pt">Download</a></td>
    </tr>
</table>

## Data Preparation

The training data is from the [BTCV challenge dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752)

- Target: 13 abdominal organs including 1. Spleen 2. Right Kidney 3. Left Kidney 4. Gallbladder 5. Esophagus 6. Liver 7. Stomach 8. Aorta 9. IVC 10. Portal and Splenic Veins 11. Pancreas 12. Right adrenal gland 13. Left adrenal gland.
- Task: Segmentation
- Modality: CT
- Size: 30 3D volumes (24 training + 6 testing)

Make sure you have a Synapse Account in order to download the RawData.zip file.
After the download is completed, extract and store the data in `/tmp/RawData/`.
To use this data, passing `--data-root` argument when training

# Train

A `SwinUNETR` network with standard hyperparameters for multi-organ semantic segmentation

```python
model = SwinUNETR(
    in_chans=1,
    num_classes=14,
    img_size=(96, 96, 96),
    embed_dim=48,
    use_checkpoint=True,
    spatial_dims=3
)
```

## Training from Self-supervised weights on single GPU (base model with gradient check-pointing)

```bash
python main.py --data-root `$(pwd)`/datasets/BTCV/data.json \
    --model-name 'SwinUNETR' --feature-size 48 \
    --loss-fn 'dice_ce' --opt-name 'adamw' --lr-scheduler 'warmup_cosine' \
    --train-cache-num 24 --valid-cache-num 6 --batch-size 1 \
    --eval-freq 100 --save-freq 100 --use-checkpoint \
    --pretrained './pretrained/SwinUNETR/swin-3d-backbone.pt' \
    --resume ''
```

## Training from scratch on multiple GPUs (base model with AMP)

```bash
torchrun --nnodes=1 --nproc-per-node=4 main.py \
    --data-root `$(pwd)`/datasets/BTCV/data.json \
    --batch-size 1 --num-classes 14 --workers 4 \
    --model-name 'SwinUNETR' --feature-size 48 \
    --distributed --amp --use-checkpoint \
    --opt-name 'adamw' --loss-fn 'dice_ce' --lr-scheduler 'warmup_cosine' \
    --train-cache-num 24 --valid-cache-num 6 \
    --eval-freq 100 --save-freq 100 \
    --pretrained '' --resume ''
```

The command above is initialized with distributed training, using gradient check-pointing and AMP.

Optional values:

- Model name: [`UNETR`, `SwinUNETR`]
- Loss function: [`dice`, `dice_ce`].
- Optimizer: [`adam`, `adamw`, `sgd`]
- LR scheduler: [`warmup_cosine`, `cosine_anneal`]

# Fine-tuning

Please download the checkpoints for models presented in the above table and place the model checkpoints in `./runs/SwinUNETR/` folder.

- To fine-tune base model on a single GPU with gradient check-pointing:

```bash
python main.py --data-root `$(pwd)`/datasets/BTCV/data.json \
    --batch-size 1 --num-classes 14 --workers 4 \
    --max-epochs=50 --use-checkpoint \
    --train-cache-num 24 --valid-cache-num 6 \
    --feature-size 48 --model-name 'SwinUNETR' \
    --resume './runs/SwinUNETR/swin_unetr.base_5000ep_lr2e-4.pt'
```

Use the same config for the small and tiny model size.

# Evaluation

pending implemented

<details>
    <summary>References:</summary>

    * [Self-supervised pre-training of swin transformers for 3d medical image analysis](https://arxiv.org/pdf/2111.14791.pdf)
    * [Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images](https://arxiv.org/pdf/2201.01266.pdf)
    * [SwinUNETR training code](https://github.com/Project-MONAI/research-contributions/blob/main/SwinUNETR/BTCV)

</details>
