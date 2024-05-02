---
layout: default
---

# Introduction

Digital writing has experienced a significant surge in popularity among students, academics, and business professionals, thanks to the data processing capabilities of tablets and the enhanced precision of styluses. Utilizing tablets for note-taking allows users to effortlessly organize, share, and search their notes, with the latter function making use of optical character recognition (OCR) technology. OCR models harness tools from computer vision and machine learning, including convolutional neural networks and vision transformers. However, there exists an intriguing outlier amidst this digital revolution – Thomas Zeng.

<div class="container">
<img src="assets/images/thomas_notability.png" width="300"/>
<figcaption> Notability fails to recognizing Thomas's hand writting. </figcaption>
</div>

Thomas relies on his iPad and the popular app Notability for note management. However, his exceptionally messy handwriting poses a unique challenge: Notability's built-in OCR model struggles to decipher his notes, rendering them unsearchable. This issue extends beyond Notability; even sophisticated OCR models falter in deciphering Thomas's handwriting. While handwriting recognition is generally considered a solved problem, we contend that our situation presents a greater challenge due to its lack of clarity – even humans find it difficult to decipher Thomas's writing.

As good friends, we took it upon ourselves to address this dilemma. Our project's primary objective became clear: to develop a model capable of accurately recognizing Thomas's handwriting. More broadly, we aim to design a handwriting recognition system that surpasses human capabilities. To achieve this ambitious goal, we explored five distinct approaches in building such a model.

# Preliminaries

## Datasets

### Thomas and Alex's handwriting

To capture Thomas's handwriting style accurately, we assembled a dataset comprising 60 images of sentences he had written. Of these, 50 images are allocated for training across our various approaches, while the remaining 10 are reserved for testing purposes. As a control, we also compiled a secondary dataset featuring sentences written by Alex, whose handwriting is notably clearer and easier to interpret.

<div class="container">
<img src="assets/images/dataset.png"/>
<figcaption> Top: example of Alex's handwriting. Bottom: example of Thomas's handwriting.</figcaption>
</div>

### IAM dataset

For training-based approaches, we also consider using IAM dataset [1]. The IAM dataset comprises an extensive collection of handwritten samples contributed by 657 individual writers. It encompasses 1,539 scanned pages of text, featuring 5,685 accurately labeled isolated sentences. This dataset has been widely used in various state-of-the-art OCR methods [2].

<div class="container">
<img src="assets/images/IAM.jpg"/>
<figcaption> Example of handwriting in IAM dataset. Figure is from [1].</figcaption>
</div>

## Evaluation metrics - character error rate

To evaluate each of our approaches we used the character error rate or CER of our predictions. CER is a common metric in natural language processing tasks and is a measure of what percent of the sentence did the model correctly predict. Because a model has the capacity to make predictions far longer than the true label, the CER can be arbitrarily large for poor predictors. For reference a good OCR model typically achieves a CER of around ~0.02. However, on Thomas' handwriting, some of these same models had a CER > 1.

<div class="container">
<img src="assets/images/cer.png"/>
<figcaption> Character Error Rate (CER). </figcaption>
</div>

## Baseline -TrOCR

In this project, we consider TrOCR [2] as our baseline. TrOCR is a transformer-based model for end-to-end text recognition. It consists of an image Transformer encoder and an autoregressive text Transformer decoder. The TrOCR model that is pretrained on IAM dataset is avalable on HuggingFace (<a src="microsoft/trocr-small-handwritten"> microsoft/trocr-small-handwritten </a>).

<div class="container">
<img src="assets/images/trocr_architecture.jpg"/>
<figcaption> Overview of TrOCR architecture. The model consists of an image Transformer encoder and an autoregressive text Transformer decoder. Figure is from [2].</figcaption>
</div>

# Methods

## Aprroach 1 - Naive Finetuning

Our first approach was to fine-tune the baseline on Thomas' handwriting with the hope that the model could adapt quickly. In order to enable more efficient fine-tuning we also ran experiments where we frooze all of the weights in the encoder while keeping the decoder weigths trainable. We also varied the number of images used in fine-tuning to see how much data was needed to adapt to a new writer. Results are shown in Table 1 and Table 2 respectively.

<div class="container">
    <table class="center" style="width: 50%; flex: 1;">
        <tr>
            <th>Finetuned on</th>
            <th>Dataset</th>
            <th>CER</th>
        </tr>
        <tr>
            <td>None</td>
            <td>Thomas</td>
            <td>5.06</td>
        </tr>
        <tr>
            <td>Entire model</td>
            <td>Thomas</td>
            <td>0.84</td>
        </tr>
        <tr>
            <td>Decoder only</td>
            <td>Thomas</td>
            <td>0.65</td>
        </tr>
        <tr>
            <td>None</td>
            <td>Alex</td>
            <td>3.10</td>
        </tr>
        <tr>
            <td>Entire model</td>
            <td>Alex</td>
            <td>0.79</td>
        </tr>
        <tr>
            <td>Decoder</td>
            <td>Alex</td>
            <td>0.34</td>
        </tr>
    </table>
    <figcaption> Table 1: Results of naive finetuning TrOCR model on Thomas and Alex's handwriting.</figcaption>
</div>

<br>

<div class="container">
    <table class="center" style="width: 50%; flex: 1;">
        <tr>
            <th># of Images</th>
            <th>Best CER (Thomas)</th>
        </tr>
        <tr>
            <td>50</td>
            <td>0.65</td>
        </tr>
        <tr>
            <td>20</td>
            <td>0.89</td>
        </tr>
        <tr>
            <td>10</td>
            <td>0.92</td>
        </tr>
        <tr>
            <td>5</td>
            <td>1.74</td>
        </tr>
        <tr>
            <td>1</td>
            <td>2.62</td>
        </tr>
    </table>
    <figcaption> Table 2: Results of varying number of labeled samples in finetuning the TrOCR model.</figcaption>
</div>

The results shown above demonstrate that Thomas' dataset is both visually more difficult for humans and quantitatively more difficult for the OCR model than Alex's dataset. We also see that freezing the encoder facilitates more efficient fine-tuning in all instances. However, the performance is still far away from what we would hope for in an OCR model.

## Aprroach 2 - Supervised domain adaptation

<div class="container">
<img src="assets/images/supervised_domain_adaptation.png" width="400px"/>
<figcaption> Figure is from [3].</figcaption>
</div>

## Aprroach 3 - Transfer learning

<div class="container">
<img src="assets/images/transfer_learning.png" width="400px"/>
<figcaption> Figure is from [4].</figcaption>
</div>

## Aprroach 4 - Dual-decoder

<div class="container">
<img src="assets/images/dual_decoder.png" width="300px"/>
<figcaption> .</figcaption>
</div>

## Aprroach 5 - Meta learning

<div class="container">
<img src="assets/images/meta_learning.png"/>
<figcaption> We split the IAM dataset into handwritings of different users and use MAML [5], a meta learning methods, to pretrain the TrOCR model. The meta learning mechanism allows our model to easily adapted to handwriting of novel users, e.g., Thomas and Alex.</figcaption>
</div>

# References

[1] <b>A full English sentence database for off-line handwriting recognition.</b> <br>
    Marti, U-V and Bunke, Horst. In Proceedings of the Fifth International Conference on Document Analysis and Recognition, 1999, pp. 705--708.

[2] <b>Trocr: Transformer-based optical character recognition with pre-trained models.</b> <br>
   Li, Minghao and Lv, Tengchao and Chen, Jingye and Cui, Lei and Lu, Yijuan and Florencio, Dinei and Zhang, Cha and Li, Zhoujun and Wei, Furu. In Proceedings of the AAAI Conference on Artificial Intelligence, 2023, pp. 13094--13102.

[3] <b>Parameter-efficient tuning on layer normalization for pre-trained language models.</b> <br>
    Qi, Wang and Ruan, Yu-Ping and Zuo, Yuan and Li, Taihao. In arXiv preprint arXiv:2211.08682, 2022.

[4] <b>Parameter-efficient transfer learning for NLP.</b> <br>
    Houlsby, Neil and Giurgiu, Andrei and Jastrzebski, Stanislaw and Morrone, Bruna and De Laroussilhe, Quentin and Gesmundo, Andrea and Attariyan, Mona and Gelly, Sylvain. In International conference on machine learning, 2019, pp. 2790--2799.

[5] <b>Model-agnostic meta-learning for fast adaptation of deep networks.</b> <br>
    Finn, Chelsea and Abbeel, Pieter and Levine, Sergey. In International conference on machine learning, 2017, pp. 1126--1135.