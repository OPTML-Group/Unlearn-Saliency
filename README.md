# SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation

<div align='center'>
<img src = 'Images/teaser-v2.png'>
</div>

With evolving data regulations, machine unlearning (MU) has become an important tool for fostering trust and safety in today's AI models. However, existing MU methods focusing on  data and/or weight perspectives often grapple with limitations in unlearning accuracy, stability, and cross-domain applicability. To address these challenges, we introduce the concept of 'weight saliency' in  MU, drawing parallels with input saliency in model explanation. This innovation directs MU's attention toward specific model weights rather than the entire model, improving effectiveness and efficiency. The resultant method that we call saliency unlearning (Salun)   narrows the performance gap with 'exact' unlearning (model retraining from scratch after removing the forgetting dataset). To the best of our knowledge, Salun is the first principled MU approach adaptable enough to effectively erase the influence of forgetting data, classes, or concepts in both image classification and generation. As highlighted below, For example, Salun yields a stability advantage in high-variance random data forgetting, e.g., with a 0.2% gap compared to exact unlearning on the CIFAR-10 dataset.  Moreover, in realm of preventing conditional diffusion models from generating harmful images,  Salun achieves nearly 100% unlearning accuracy.

## Getting started
SalUn can be applied to different tasks such as image classification and image generation. You can click the link below to access a more detailed installation guide.
* [SalUn for Image Classification](Classification/README.md)
* SalUn for Image Generation
  * [DDPM](DDPM/README.md)
  * [Stable Diffusion](SD/README.md)

## Contributors

* Chongyu Fan
* [Jiancheng Liu](https://scholar.google.com/citations?user=ReWNzl4AAAAJ&hl=en)
