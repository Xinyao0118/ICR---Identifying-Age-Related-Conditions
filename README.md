# Google - American Sign Language Fingerspelling Recognition
Train fast and accurate American Sign Language fingerspelling recognition models

![competition](img/bgr.png)

[Google - American Sign Language Fingerspelling Recognition](https://www.kaggle.com/competitions/asl-fingerspelling)

## Competition Introduction
The goal of this competition is to detect and translate American Sign Language (ASL) fingerspelling into text. 
You will create a model trained on the largest dataset of its kind, released specifically for this competition. 
- Type: Automated Speech Recognition (ASR), NLP
- Recommended Model: TensorFlow Lite model
- Evaluation Metrics: [levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
- Baseline: 

## Data Introduction
Files:
124 files

Size:
189.09 GB

Type:
parquet, csv, json
### **[train/supplemental_metadata].csv** 
### character_to_prediction_index.json
### [train/supplemental]_landmarks/


## Solution Approach

- Preprocessing - Tensorflow
- Embedding (Landmark Embedding + Embedding)
- Transformer - replace softmax with softmax layer to support masked softmax
- Adam Optimizer

## Updates

- Aug 2: Base model: CV 1.77 | **LeaderBoard 386/1119 | TOP 35%**
- 

## Core code 

```Python

```

## Conclusion

