<!---
layout: default
-->

# Problem Statement

## Background
Digital writing has seen a signficant rise amongst students, academics, and buisness professionals due to the data process capabilities of tablets and imrpoved precision on styluses. Taking notes on a tablet provides allows users to easily organize, share, and search their notes. The search function in particular makes use of a technology called optical character recognition or OCR. OCR models leverage tools from computer vision and machine learning such as convolutional neural network and vision transformers. Furthermore, these methods can acurately classify a user's handwritting... unless your name is Thomas Zeng. 

<!--- ![Alt text](thomas_notability.png "We're in for a challenge") -->
<style>
  .padded-image {
    padding: 10px; /* Adjust the padding value as needed */
  }
</style>

<img src="thomas_notability.png" alt="Alt text" align="right" class="padded-image" width="300"/>

<!--- <style>
  .image-container {
    display: flex; /* Use flexbox layout */
    flex-direction: column; /* Stack elements vertically */
    align-items: center; /* Center items horizontally */
  }
  .padded-image {
    padding: 10px; /* Adjust the padding value as needed */
  }
  .title {
    margin-top: 5px; /* Add some space between the image and the title text */
  }
</style> 

<div class="image-container">
  <img src="thomas_notability.png" alt="Alt text" class="padded-image" width="300"/>
  <div class="title">Title text</div>
</div>-->

## Project Goal
Thomas uses his iPad and a popular app called notability to manage the notes he takes. However, because of his unusually messy handwriting, notability's built in OCR model cannot recognize his handwriting meaning he cannot search his notes. This problem isn't specific to notability, other sophisticated OCR models also fail to recognize Thomas' handwriting. While handwriting recognition is considered to be a solved problem, we argue that our problem is more difficult because it is not “well defined”, in the sense that it is hard for humans to recognize what Thomas writes.
Being the good friends we are, we made our project goal to create a model that could recognize his writing along. More broadly, our goal is to design a handwriting recongition system that superseeds human capabilities.

## Datasets
To represent Thomas' handwriting, we created a dataset of 60 images of sentences he had written. We used 50 of these images across our different training approaches and 10 for testing. As a control we create a second dataset comprised of Alex's handwriting for the same sentences which is much easier to read. 

<img src="dataset.png" alt="Alt text" align="right" class="padded-image" width="500"/>

We wanted to keep the datasets we created relatively small to limit the amount data a user with messy handwriting would need to provide to get an adapted model. However, some of our methods required a larger volume of data along with text from a large pool of writers. For these purposes, we used the IAM dataset, which is composed of ~10,000 lines of text from hundreds of unique writers.

## Evaluation Metrics

## Approch 1


## Approch 2

## Approch 3

## Approch 4

## Approch 5

## References
1. IAM?
2. TrOCR
more tex2t more tex2tmore tex2tmore tex2tmore tex2tmore tex2tmore tex2tmore tex2tmore tex2tmore tex2tmore tex2t
