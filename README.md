# non-vae-ldm
Codebase for my research of Latent Difuusion Models with non-vae encoders.


## Getting stargted

I used `python 3.10.12`

To start from root of repository run command

```bash
./setup/setup.sh
```

It will create virtual env with necessary libraries, downnload checkpoints and datasets from public yandex disk

## Experiments

activate virtual env

```bash
source venv_viz/bin/activate
```
### Interpolation sanity check

In this section we repeat experiment from original svg article on interpolations from noise and following denoising.

Results example output could be seen in file `interpolation_sanity_check/results.ipynb`

to run experiment:

```bash
venv_vis/bin/python interpolation_sanity_check/run.py {path/to/config}
```

instead of `venv_vis/bin/python` you can use just `python`, if venv_vis environment already activated.


`path/to/config` is optional, you can specify your own, or not, then `configs/interpolation_sanity_check_base.yaml` will be used.




### Separation Spot detecting

This experiment creates images, that were sampled using algorythm of denoising under 2 different labels (we check switch from label 1 to 2 at different timestamps).

Pusrpose is to detect, when classes are separated, so while model is in field of label 1, and cannot rich field of label 2.

Results example output could be seen in file `separation_spot/results.ipynb`

to run experiment:

```bash
python separation_spot/run.py {configs/separation_spot_base.yaml}
```

you can specify your `path/to/config`, or not, then `configs/separation_spot_base.yaml` will be used.