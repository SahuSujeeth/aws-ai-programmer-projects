# AWS AI Programmer Projects

Two projects from the **Future AWS AI Programmer Nanodegree** (AWS AI & ML Scholars program with Udacity, 2026).

> **Note:** I built these projects with help from an AI assistant, and I reviewed and tested the code. The starter files and the `classifier.py` helper were provided by Udacity.

---

## 1. Dog Breed Image Classifier (`dog-breed-classifier/`)

A Python program that uses pre-trained CNN models to find out if a pet image is a dog, and which breed it is. It compares three models: **VGG**, **AlexNet** and **ResNet**.

**What the program does**
- Reads command line inputs (image folder, model, dog names file)
- Makes pet labels from the image file names
- Classifies every image with the chosen CNN
- Checks which labels are dogs, using `dognames.txt`
- Calculates the results and prints a summary, with the run time

**Results on 40 images (VGG)**
| Measure | Result |
|---|---|
| Dogs found correctly | 100% |
| Not-dogs found correctly | 100% |
| Breeds correct | 93.3% |
| Labels matched | 87.5% |

**Extra test with my own 4 images:** all three models told a dog from a cat and a coffee mug correctly. When I rotated the same dog photo by 180 degrees, AlexNet still said golden retriever, but VGG and ResNet said chihuahua.

**How to run**
```bash
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
```
The `pet_images` folder is part of the Udacity course, so it is not included here.

---

## 2. Movie Review Sentiment Analysis (`sentiment-analysis/`)

A small **transformer** model, written in PyTorch, that reads an IMDB movie review and says if it is positive or negative.

**Details**
- Data: IMDB reviews (25,000 for training, 25,000 for testing). I split the training set into 22,500 for training and 2,500 for validation.
- Tokenizer: `bert-base-uncased`, maximum length 128 tokens
- Model: 4 transformer layers, 4 attention heads, embedding size 128
- Change for classification: mean pooling over the token outputs, then a linear layer with 2 outputs
- Training: 3 epochs, AdamW, learning rate 3e-4

**Results**
| Epoch | Validation accuracy |
|---|---|
| 1 | 72.00% |
| 2 | 75.12% |
| 3 | 76.92% |

**Test accuracy: 76.38%**

The notebook is `SentimentScope_starter.ipynb`. The dataset is the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/) and is not included here.

---

## Tools
Python, PyTorch, Hugging Face Transformers, pandas, matplotlib
