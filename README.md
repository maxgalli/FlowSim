# FlowSim
Code for training the models of the paper: xxx

## Generate data

To generate 100k events (about 500k jets) for training on a ttbar dataset, run data/generator.py
```
  python generator.py 100000 "gen_ttbar_100k_.npy" --ttbar --seed=$seed 
```
NOTE: Pythia and FastJets must be correctly installed

## Train model

To train the best model from the paper, run the src/train_cfm.py with the provided config:

```
python train_cfm.py CRT.yaml
```

the python packages required are listed under requirements.txt

## Acknowledgements
Contains code inspired by [dingo](https://github.com/dingo-gw/dingo/tree/FMPE), [nflows](https://github.com/bayesiains/nflows) and [torchcfm](https://github.com/atong01/conditional-flow-matching) code bases, released under MIT License
