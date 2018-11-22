# Parametric generation of conditional geological realizations using generative neural networks ([arXiv](https://arxiv.org/abs/1807.05207))

*Requires PyTorch 0.4+*

Run with `python main.py [--options]`

<!-- ![img1](https://i.imgur.com/p7SWQCn.png)
![img2](https://i.imgur.com/S2aWSo1.png)
![img3](https://i.imgur.com/LfZCmwg.png) -->

- Z
<img src=https://i.imgur.com/p7SWQCn.png width=500 />

- X
<img src=https://i.imgur.com/S2aWSo1.png width=500 />

- O
<img src=https://i.imgur.com/LfZCmwg.png width=500 />

You can download our pre-trained unconditional generator `netG.pth` [here](https://drive.google.com/file/d/1E7Rm2Fao3RJ3fQnmd8csWQVlZ6-6cEtx/view?usp=sharing) (12MB), which has been trained using Wasserstein GAN ([https://arxiv.org/abs/1701.07875](https://arxiv.org/abs/1701.07875))

- `main.py` main code
- `models.py` neural network architectures
- `utils.py` helper functions
- `harddata*.dat` conditioning test cases
