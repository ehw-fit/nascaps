# NAScaps
A Framework for Neural Architecture Search to Optimize the Accuracy and Hardware Efficiency of Convolutional Capsule Networks. This repository provides source codes and [the best-found solutions](chroms/README.md) from neural architecture search of convolutional capsule networks. For more detail please follow [a paper](https://arxiv.org/abs/2008.08476). If you used these results in your research, please refer to the paper


     MARCHISIO Alberto, MASSA Andrea, MRAZEK Vojtech, BUSSOLINO Beatrice, MARTINA Maurizio and SHAFIQUE Muhammad. NASCaps: A Framework for Neural Architecture Search to Optimize the Accuracy and Hardware Efficiency of Convolutional Capsule Networks. In: IEEE/ACM International Conference on Computer-Aided Design (ICCAD '20). Virtual Event: Institute of Electrical and Electronics Engineers, 2020, p. 9. ISBN 978-1-4503-8026-3. 


```bibtex
@INPROCEEDINGS{nascaps:2020,
   author = "Alberto Marchisio and Andrea Massa and Vojtech Mrazek and Beatrice Bussolino and Maurizio Martina and Muhammad Shafique",
   title = "NASCaps: A Framework for Neural Architecture Search to Optimize the Accuracy and Hardware Efficiency of Convolutional Capsule Networks",
   pages = 9,
   booktitle = "IEEE/ACM International Conference on Computer-Aided Design (ICCAD '20)",
   year = 2020,
   location = "Virtual Event, US",
   publisher = "Institute of Electrical and Electronics Engineers",
   ISBN = "978-1-4503-8026-3",
   doi = "10.1145/3400302.3415731",
   url = "https://arxiv.org/abs/2008.08476"
}
```

If you found any problem or something in the description is not clear, please feel free to create an issue ticket.

## Installation of the testing environment

The testing environment requires TensorFlow. Moreover, a Pareto-frontier package must be installed.

    pip3 install --user git+https://github.com/ehw-fit/py-paretoarchive


### Instalation using Anaconda
```bash
conda env create -n tf-1.13-gpu -f environment.yml


source activate tf-1.13-gpu
pip install --user git+https://github.com/ehw-fit/py-paretoarchive

# do the commands
conda deactivate
```



## Using a scripts

All executable scripts are located in "nsga" folder. You can use `-h` parameter to get a list of available parameters. Please note that not all parameters must be used in the scripts. See examples bellow

 * `main.py` runs NSGA NAS search 
 * `randlearn.py` generates random neural networks and evaluates them
 * `chreval.py` evaluates the results from NAS for a larger amount of epochs
 * `chreval_complex.py` a different training algorithm optimized for CIFAR-10 for evaluation

### Training 

```bash 
work="results" # working directory for saving results

source activate tf-1.13-gpu # activating valid environment with Anaconda

cd nsga

dat="mnist-$(date +"%F-%H-%M")"
python main.py --epochs 5 \   # number of epochs
	--output out_${dat} \     # 
	--population 10 --offsprings 10 \  #nsga settings
	--generations 50 \   # 
        --timeout 300 \  # timeout in in seconds for training of one candidate
	--save_dir ${work}/data \  # directory with results
	--cache_dir ${work}/cache \ # 
	2>${work}/logs/${dat}.err > ${work}/logs/${dat}.std
```


```sh
python randlearn.py --epochs 10 \
	--dataset cifar10 \
	--output "${work}/rand_short/cifar/${dat}" \
	--max_params 0 \
        --timeout 600 \
	--save_dir ${work}/data \
	--cache_dir ${work}/cache \
	2>${work}/rand_short/cifar/${dat}.err > ${work}/rand_short/cifar/${dat}.std &
```


### Testing
```sh
cd nsga
python chrlearn.py --epochs 150 \
	--output "./logrand/${dat}" \
	--dataset mnist \
	--max_params 0 \
	"$fn"                       \
	2>../logrand/${dat}.err > ../logrand/${dat}.std
```

```sh
ds="cifar100"
PARAMS="--batch_size 100 --epochs 300 --shift_fraction 0.078125 --rotation_range 2 --horizontal_flip --lr 0.001 --lr_decay 0.96 --decay_steps 6000 --lam_recon 0.005 "
python chrlearn_complex.py --dataset ${ds} --epochs 300 $PARAMS ../data/chrom_deepcaps.chr 1>${outdir}/${ds}-deepcaps.std.log 2>${outdir}/${ds}-deepcaps.err.log 
```
