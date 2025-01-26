# SSSR (Speech Stress State Recognition)

A project to evaluate stress levels from speech data using AI. This work involves mapping emotional speech data to stress levels through various deep learning models.

## Description

SSSR focuses on analyzing speech data to predict stress states. By leveraging emotional speech datasets such as **TESS** and **CREMA-D**, this project aims to map audio features to corresponding stress levels. The model development includes a range of approaches, from basic CNN architectures to advanced transformer-based models.

### Models Used:
- **CNN**: Custom convolutional neural network tailored for audio feature extraction.
- **Transformer**: Sequence-based transformer model for temporal pattern recognition in audio signals.
- **1×128 Vertical Patch Vision Transformer**: A specialized transformer variant for fine-grained feature extraction.
- **HuBERT (LS-960) Finetuning**: Finetuned HuBERT model leveraging pre-trained self-supervised audio representations.

### Current Progress:
The project is in its early stages and is actively being developed. While preliminary experiments have been conducted on **TESS** and **CREMA-D**, we are awaiting access to the **BESST dataset**, a speech-stress-specific dataset, to enhance model performance and validate results.

## Datasets

### [Toronto Emotional Speech Set (TESS)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
A dataset consisting of emotional speech samples recorded by two actors, covering seven different emotions.

### [CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)](https://www.kaggle.com/datasets/ejlok1/cremad)
A dataset containing emotionally diverse audio recordings performed by 91 actors with six basic emotions and neutral expressions.

### BESST Dataset
The **BESST dataset**, designed specifically for mapping speech to stress levels, is a key component for the next phase of development. Access to this dataset is currently pending.

## Future Plans
- Fine-tune models on the BESST dataset to achieve higher accuracy and robustness.
- Expand the dataset pool for broader applicability.
- Experiment with additional self-supervised audio models for improved generalization.

## My first Deep Learning Project
Sorry, It's little messy.
