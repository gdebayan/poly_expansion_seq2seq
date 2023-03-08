# Polynomial Expansion using Seq2Seq Transformer Encoder-Decoder Network

This repository implements a  [Sequence-to-Sequence Transformer model](https://arxiv.org/pdf/1706.03762.pdf) for Polynimial Expansion.

## Introduction
The nature of the problem (Polynomial Expansion) is synonymous with Seq2Seq tasks such as Automatic Speech Recognition (ASR), Machine Translation, etc. Traditionally [LSTM/RNN recurrence models](https://arxiv.org/abs/1508.01211) were employed for such tasks, where the Encoder Hidden States were used to capture Input information, and an Attention-Based Auto-regrssive Decoder was used as a Language model to generate new text.

However, more recent work on [Transformers](https://arxiv.org/pdf/1706.03762.pdf) built on Self-Attention Layers showed an improvement in Seq2Seq tasks 
when compared to traditional recurrent networks. Motivated by this, the Transformer model was chosen for the Polynomial Expansion task.

## Model Architecture

The model layer configuration chosen for this task are the following:

* Number of Encoder Layers: 4
* Number of Decoder Layers: 4
* Number of Attention Heads: 8
* Hidden Dimension: 256

The number of Trainable Parameters in the model: **4,249,636**. Refer [network.txt](network.txt) for more details.

NOTE: The following parameters can be configured from [config.py](config.py)

## Hyperparameter Choices

* Optimizer:  We used the Adam optimizer, with 
    * initial **Learning Rate = 0.0001**. 
    * The value of **Beta_1 (Running dLdW) = 0.9**
    * The value of **Beta_2 (Running dLdW\*\*2) was 0.98**
    
We did hypermarameter tuning to arrive at these parameters. 
Adam was chosen as the optimizer, as it combines **Momentum** and **RMSProp** optimization methods, i.e. 

    1) Mometum Property: The derivative update at step **t** is a running average of the current derivative, and past derivative values (Momentum).
      This helps us take steps on an average of current and past gradients, and helps in smoother convergence
    2) RMSProp Property: RMSProp estimates the Hessian, and this enables each dimension to have an independent learning rate
    
* Scheduler: We employed the ReduceLR on Pleateau scheduler.
    
## Data Preparation

The full dataset can be found at [dataset_files/dataset.txt](dataset_files/dataset.txt), which has 1,000,000 samples
* **Train Split = 0.93** : [Train (train.txt)](dataset_files/train.txt) with 930,000 samples
* **Validation Split = 0.01** : [Val (val.txt)](dataset_files/val.txt) with 10,000 samples
* **Test Split = 0.06** : [Test (test.txt)](dataset_files/test.txt) with 60,000 samples

The reasoning for the split is as follows:
* We wanted a significant amount of Test Samples (60,000) to proove our model actually works.
* For Validation, 10,000 samples was sufficient to track if the model was learning apprporiately (i.e track overfitting, etc.)
* For Training, 930,000 samples were chosen which was sufficient for training the model.
  
**NOTE:** Training with 980,000 samples resulted in significant improvent in Test Accuracy (On 10,000 test samples). However, I felt having a slightly lower accuracy **(~4%)** on 60,000 test samples was a better indication of model performance, than having a higher accuracy with 10,000 test samples

## Tokens
The following was the vocabulary for our model. This was obtained by analysing the Train dataset.

**sin, cos, tan, numeric digit, alphabet, (, ), +, -, \*\*, \***.

Refer to [data_utils/polynomial_vocab.py](data_utils/polynomial_vocab.py) for more details


## Reproduce

* **Change to Working Directory**: `cd polynomial_expansion_seq2seq/`
* **Setup**: `bash setup.sh`
* **Running Evaluation only**: `python3 eval_main.py`. 
  By default, [checkpoints/val_best_ref.pth](checkpoints/val_best_ref.pth) is set to `MODEL_PATH` in [eval_main.py](eval_main.py).
  This .pth filed can be changed if evaluation on a new model is desired.
* **Running Training**: `python3 train_main.py`.
  The checkpoints **(model_epoch_x.pth)**, **val_best.pth** will be stored at [checkpoints/](checkpoints/) folder.
  We can configure the last **N** checkpoints to retain by setting **NUM_LATEST_CHECKPOINT_SAVE** (Default=4) in [config.py](config.py)
* **Running Training with new Dataset generation**: `python3 train_main.py`, Set **RECOMPUTE_DATASET=True** in [config.py](config.py)

## Examples

```
#1: 
 Input: (h-2)*(5*h+28) 
 Expected: 5*h**2+18*h-56 
 Predicted Output: 5*h**2+18*h-56
 
#2: 
 Input: (-4*k-25)*(2*k+18) 
 Expected: -8*k**2-122*k-450 
 Predicted Output: -8*k**2-122*k-450
 
#3: 
 Input: (26-3*cos(i))*(-5*cos(i)-28) 
 Expected: 15*cos(i)**2-46*cos(i)-728 
 Predicted Output: 15*cos(i)**2-46*cos(i)-728
 
#4: 
 Input: 20*h**2 
 Expected: 20*h**2 
 Predicted Output: 20*h**2
 
#5: 
 Input: (tan(j)-9)*(8*tan(j)+7) 
 Expected: 8*tan(j)**2-65*tan(j)-63 
 Predicted Output: 8*tan(j)**2-65*tan(j)-63

```

### Results:
| Train/Test/Val Split | Test Accuracy |
|----------------------|---------------|
| 0.93/ 0.06/ 0.01     | 92.71 %       |
| 0.98/ 0.01/ 0.01     | 96.51 %       |



