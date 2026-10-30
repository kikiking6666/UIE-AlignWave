# UIE-AlignWave

## News

🎉 Our paper, **"UIR-WaveAlign: Long-Range Anisotropic Receptive Field Alignment for High-Frequency Haar Wavelet Sub-bands in Underwater Image Restoration"**, has been accepted for publication in *The Visual Computer*!

The official paper link will be updated once available.

## ⚙️ Usage

### Training

You may download the dataset first, and then specify `TRAIN_DIR`, `VAL_DIR` and `SAVE_DIR` in the `TRAINING` section of `config.yml`.

For single GPU training:

```bash
python train.py
```

### Inference

Please first specify `TRAIN_DIR`, `VAL_DIR` and `SAVE_DIR` in the `TESTING` section of `config.yml`.

```bash
python test.py
```

## Acknowledgements

We thank the authors of [UIR-PolyKernel](https://github.com/CXH-Research/UIR-PolyKernel) for publicly releasing their code and related resources. Their codebase, dataset resources, and evaluation benchmarks provided valuable support for the implementation and experimental evaluation of our work.