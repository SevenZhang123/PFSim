# PFSim
_The code, data, and tools for our paper: "From a Global Perspective: Highly Applicable and Robust IPv4-IPv6 Address Association Method via Port Fingerprint"_:nerd_face:

## Environment Preparation
Before using this method, you need to install the following packages in advance:
```
 numpy  1.26.4+
 scikit-learn   1.5.1+
 tqdm   4.66.5+
 Pytorch    2.6.0
```

## :warning:Notice
(1). We have fine-tuned DStack-Tokenizer and DStack-BERT (in the "model" folder), which are perfect for IPv4-IPv6 address association tasks. If you want to use our tools directly, please skip to step 3 directly:white_check_mark:.  
If you want to fine-tune these two tools again, please follow steps 1-3 in order.  
(2). Module 1:"**Port Detection and Service Information Acquisition**" can be directly implemented based on Censys API, and the relevant code will not be shown in detail. 
Here we only introduce the reproduction methods of the three core modules (DStack-Tokenzier creation, DStack-BERT fine-tuning and port fingerprint construction and similarity calculation).

## Step-1 DStack-Tokenzier Creation
We build service information corpus based on training set data $D_T$ (see the paper for details), and expand the original tokenizer based on BPE algorithm to create a domain-specific tokenizer **DStack-Tokenizer**.  
Please replace the input data with your corpus and run the following code:
```
cd code
 python DStack-Tokenizer.py 
```

## Step-2 DStack-BERT Fine-tuning
This part includes two steps: **_step2-1_**: constructing training sample pairs based on data augmentation and **_step2-2_**: fine-tuning the BERT model based on **sim**ple **s**entence **c**ontrast **l**earning (**SimSCL**).

### Step2-1 Constructing Training Sample Pairs Based on Data Augmentation
We're constructing training sample pairs based on service information corpus.
Please replace our corpus ("train_info_month4.txt") with your corpus and run the following code:
```
cd code
 python data_augment.py 
```

### Step2-2 Fine-tuning BERT Model
Please replace the tokenizer and training corpus with your own (hyperparameters can also be modified as needed), and then run the following code:
```
cd code
 python SimSCL.py 
```

## Step-3 IPv4-IPv6 Address Association based on Port Fingerprint Similarity
This part includes two steps: port fingerprint construction and similarity calculation. 
You can easily run it with one click through the following code!:rocket:  
[!NOTE]
If you built the tokenizer and encoder yourself, you need to replace DStack-Tokenizer and DStack-BERT in the code.
```
cd code
 python SimFig.py 
```

[!IMPORTANT]
Due to our data usage agreement with Censys, we are unable to provide our detection results for each target IP, such as port open status and service information.
We recommend that users first apply for data access permissions from Censys or other institutions, and then use our method after obtaining relevant data.
