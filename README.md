# DDSP-Violin

Code accompanying **"DDSP-Violin: Physically-Informed Constraints for Disentangled Source-Filter Decomposition"** (EUSIPCO 2026).

A physically-informed DDSP framework for bowed-string synthesis. Constrains the harmonic source with low-dimensional parameters inspired by bowed-string acoustics (brightness $\alpha$, bow position $\beta$, notch depth $\gamma$, residuals $\rho_n$) to improve source-filter disentanglement.

## Audio examples

Listen at [joaoerduarte.github.io/ddsp_violin](https://joaoerduarte.github.io/ddsp_violin/).

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py                  # DDSP-Violin
python train.py --CONFIG ddsp    # original DDSP
```

Audio: 16 kHz mono `.wav` files in `dataset/audio/`. Preprocessing: `python preprocess.py`.

## Citation

```bibtex
@inproceedings{duarte2026ddspviolin,
  title={{DDSP-Violin}: Physically-Informed Constraints for Disentangled Source-Filter Decomposition},
  author={Duarte, Jo{\~a}o and Mignot, R{\'e}mi and McDermott, James and O'Leary, Se{\'a}n},
  booktitle={Proc. EUSIPCO},
  year={2026}
}
```

## Acknowledgments

Built upon [acids-ircam/ddsp_pytorch](https://github.com/acids-ircam/ddsp_pytorch). Original DDSP framework: Engel et al. (ICLR 2020). Bowed-string physical model: Demoucron (2008).

Supported by the Research Ireland Centre for Research Training in Digitally-Enhanced Reality (d-real), Grant No. 18/CRT/6224.

## License

MIT
