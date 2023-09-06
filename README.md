<!-- omit in toc -->

# awesome-LLMs [![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)

A compilation of AWESOME things of Large Language Models (LLMs) is presented. Each LLM is detailed through a structured summary, highlighting shared characteristics for easy comparison. This structured approach aligns with the next-generation scholarly communication standards, allowing seamless integration into any Scholarly Knowledge Graph (SKG). A prime example is the machine-actionable [review article](https://orkg.org/review/R609546) on the Open Research Knowledge Graph ([ORKG](https://orkg.org/)) or the [comprehensive comparison](https://orkg.org/comparison/R609337/) of LLMs.


## Contents

- [awesome-LLMs](#awesome-LLMs-)
  - [Organizations](#Organizations)
	- [OpenAI](#openai)
	- [Google](#google)
	- [Google, and CMU](#google-and-cmu)
	- [Pengcheng Lab, and Baidu](#pengcheng-lab-and-baidu)
	- [Google, and University of Washington](#google-and-university-of-washington)
	- [Salesforce](#salesforce)
	- [Deepmind](#deepmind)
	- [Facebook](#facebook)
	- [Microsoft](#microsoft)
	- [Huggingface](#huggingface)
	- [Google, and Imperial College London](#google-and-imperial-college-london)
	- [Google, and Stanford](#google-and-stanford)
	- [NVidia](#nvidia)
	- [EleutherAI](#eleutherai)
	- [Facebook, Google, and UC Berkeley](#facebook-google-and-uc-berkeley)
	- [UC Berkeley](#uc-berkeley)
	- [AI21](#ai21)
	- [Anthropic](#anthropic)
	- [EleutherAI, Stability.ai, and LMU Munich](#eleutherai-stability.ai-and-lmu-munich)
	- [Amazon](#amazon)
	- [Tsinghua University](#tsinghua-university)
	- [BigScience](#bigscience)
	- [Huggingface, and Big Science](#huggingface-and-big-science)
	- [Meta](#meta)
	- [Meta AI, University of Washington, and University of Hong Kong](#meta-ai-university-of-washington-and-university-of-hong-kong)
	- [Meta AI](#meta-ai)
	- [Stanford University](#stanford-university)
	- [Large Model Systems Organization](#large-model-systems-organization)
	- [MosaicML](#mosaicml)
	- [BigCode Project](#bigcode-project)
	- [Technology Innovation Institute](#technology-innovation-institute)
	- [Berkeley AI Research](#berkeley-ai-research)
	- [Cerebras, Mohamed bin Zayed University of Artificial Intelligence, and Inception](#cerebras-mohamed-bin-zayed-university-of-artificial-intelligence-and-inception)


## Organizations

#### OpenAI

- GPT-1

  ```yaml
  Title: Improving Language Understanding by Generative Pre-Training
  model family: GPT
  date created: 2018-06-01
  organization: OpenAI
  innovation: The paper introduces a framework for natural language understanding by first using generative pre-training on a diverse corpus and then fine-tuning for specific tasks. This approach improved state-of-the-art results on 9 out of 12 datasets, highlighting the potential of unsupervised learning combined with discriminative tasks.
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  fine-tuning task: Supervized discriminative finetuning
  training corpus: BookCorpus, Supervised Finetuning on several task-specific datasets for Natural Language Inference, Question Answering, Sentence similarity, and Classification.
  optimizer: Adam optimizer
  tokenization: byte pair encoding
  number of parameters: 117M
  maximum number of parameters (in million): 117
  application: Text generation, but adaptable to many other NLP tasks when fine tuned.
  has source code: https://github.com/openai/finetune-transformer-lm, https://huggingface.co/docs/transformers/model_doc/openai-gpt
  blog post: https://medium.com/@hyponymous/paper-summary-improving-language-understanding-by-generative-pre-training-7b77babd7086, https://www.reddit.com/r/MachineLearning/comments/n36htr/p_gpt1_annotated_paper_paper_summary/
  license: closed source
  research problem: Large Language Models (LLMs), transformer model
  ```

- GPT-2

  ```yaml
  Title: Language models are unsupervised multitask learners
  model family: GPT
  date created: 2019-02-01
  organization: OpenAI
  innovation: It can generate upto 768 words (equivalent to 1 1/2 page), demonstrate that language models begin to learn tasks such as question answering, machine translation, reading comprehension, and summarization without any explicit supervision when trained on a task-agnostic, diverse dataset of millions of web-scraped webpages. The work proposes a central research question: do WebText LMs transfer well across domains and datasets?
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  training corpus: https://github.com/openai/gpt-2-output-dataset, 8 million web pages (40 GB). 10X GPT . WebText dataset is created by crawling all links at Reddit with at least 3 Karma points.
  tokenization: byte pair encoding
  number of parameters: 124M, 355M, 774M, 1.5B
  maximum number of parameters (in million): 1500
  extension: GPT-2 is a direct scale-up of GPT, with more than 10X the parameters and trained on more than 10X the amount of data., Minor extensions to the GPT architecture (e.g. layer normalization moved to the input of each sub-layer, or increased context size from 512 to 1024)
  application: Text generation, but adaptable to many other NLP tasks when fine tuned.
  has source code: https://huggingface.co/docs/transformers/model_doc/gpt2
  blog post: https://openai.com/research/better-language-models, https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface
  license: closed source
  research problem: Large Language Models (LLMs), transformer model
  ```

- GPT-3

  ```yaml
  Title: Language Models are Few-Shot Learners
  model family: GPT
  date created: 2020-05-01
  organization: OpenAI
  innovation: GPT-3's primary innovation in the context of Large Language Models is its exceptional few-shot learning capabilities, allowing it to make accurate predictions using just a natural language prompt and a few task demonstrations. The model also introduced prompt-based and in-context learning methodologies. However, its vast size (175B parameters) poses challenges for real-world applications., It can generate upto 1,536 words (equivalent to 3 pages)
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  training corpus: ~ 500B tokens including CommonCrawl (410B), WebText2 (19B), Books1 (12B), Books2 (55B), and Wikipedia (3B)
  number of parameters: 125M, 350M, 774M, 1.3B, 2.7B, 6.7B, 13B, 175B
  maximum number of parameters (in million): 175000
  hardware used: Nvidia V100 GPU
  hardware information: All models were trained on V100 GPU’s on part of a high-bandwidth cluster provided by Microsoft.
  extension: Same as GPT-2 with the only addition of alternating dense and locally banded sparse attention patterns, inspired by the Sparse Transformer
  application: Initially text generation, but has over time been used for a large range of applications in areas such as code generation, but also image and audio generation
  has source code: https://platform.openai.com/docs/models/gpt-3-5, https://github.com/openai/gpt-3
  blog post: https://medium.com/analytics-vidhya/openai-gpt-3-language-models-are-few-shot-learners-82531b3d3122, https://openai.com/blog/gpt-3-apps
  license: closed source
  research problem: Large Language Models (LLMs), transformer model
  ```

- DALL-E

  ```yaml
  Title: Zero-Shot Text-to-Image Generation
  model family: GPT
  date created: 2021-01-01
  organization: OpenAI
  innovation: The paper introduces a model with remarkable generalization, capable of creatively interpreting and combining unusual textual concepts into images. It also demonstrates combinatorial generalization and zero-shot image-to-image translation controlled by natural language, showcasing advancements in LLMs for text-to-image synthesis.
  pretraining architecture: Decoder
  pretraining task: Caption prediction
  training corpus: 250 million text-images pairs from the internet
  optimizer: Adam optimizer
  tokenization: BPE-ecnode
  number of parameters: 12B
  maximum number of parameters (in million): 12000
  hardware used: NVIDIA V100 (16GB) GPU
  extension: A differential variational auto-encoder is used to learn the visual codebook. The transformer is a variation of GPT-3
  application: Text to image
  has source code: https://github.com/openai/DALL-E, https://github.com/borisdayma/dalle-mini
  blog post: https://openai.com/blog/dall-e/, https://ml.berkeley.edu/blog/posts/dalle2/
  license: N/A
  research problem: Large Language Models (LLMs), transformer model
  ```

- CLIP

  ```yaml
  Title: Learning Transferable Visual Models From Natural Language Supervision
  model family: Also using Resnet, ViT, and vanilla transformer for text, CLIP
  date created: 2021-02-01
  organization: OpenAI
  innovation: CLIP, in the context of Large Language Models, introduces a novel approach by leveraging natural language supervision with a dataset of 400 million (image, text) pairs. It excels in zero-shot learning, allowing it to classify images using textual descriptions without prior training on specific categories. This integration of vision and language offers a flexible, scalable solution with potential for diverse applications.
  pretraining architecture: Encoder
  pretraining task: predict which of the N × N possible (image, text) pairings across a batch actually occurred
  training corpus: WIT (WebImageText) - 400 million text,image pairs
  optimizer: Adam optimizer
  hardware used: Nvidia V100 GPU
  hardware information: The largest ResNet model, RN50x64, took 18 days to train on 592 V100 GPUs while the largest Vision Transformer took 12 days on 256 V100 GPUs.
  extension: Combines Resnet and ViT for the visual encoding with Transformer for the Textual encoder
  application: Image/Object classification
  has source code: https://github.com/openai/CLIP, https://huggingface.co/docs/transformers/model_doc/clip
  blog post: https://openai.com/research/clip, https://medium.com/axinc-ai/clip-learning-transferable-visual-models-from-natural-language-supervision-4508b3f0ea46
  license: Open, MIT license
  research problem: Large Language Models (LLMs), transformer model
  ```

- GLIDE

  ```yaml
  Title: GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models
  model family: Diffusion models
  date created: 2021-12-01
  organization: OpenAI
  innovation: The paper introduces two guidance techniques for text-guided image synthesis: CLIP guidance and classifier-free guidance. Of the two, classifier-free guidance produces higher-quality, photorealistic images that align closely with textual descriptions, outperforming previous models like DALL-E in evaluations.
  pretraining architecture: Encoder
  pretraining task: Caption prediction
  training corpus: Same as DALL-E
  number of parameters: 3.5B diffusion model (2.3B for visual encoding, 1.2B for textual) + 1.5B for model for upsampling
  maximum number of parameters (in million): 3500
  extension: GLIDE can be seen as an extension of the ADM (Ablated Diffusion Model) by the same authors. However, ADM is not per se a transformer architecture although it does resemble one in some of the configurations the authors use. Given that ADM is by the same authors and was quickly followed up by GLIDE, I think it is fair to consider GLIDE as the first of its kind.
  application: Text to image
  has source code: https://github.com/openai/glide-text2im
  license: Open, MIT license
  research problem: Large Language Models (LLMs), transformer model
  ```

- InstructGPT

  ```yaml
  Title: Training language models to follow instructions with human feedback
  model family: GPT
  date created: 2022-01-01
  organization: OpenAI
  innovation: Better alignment of LLMs with human expectations using reinforcement learning through human feedback
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  fine-tuning task: Reinforcement Learning from Human Feedback
  training corpus: Same as GPT3 for pretraining, but finetuned and optimized using labeler data and prompts
  number of parameters: Same as GPT3
  extension: GPTInstruct starts off with a pretrained GPT3 model and adds reward modeling through reinforcement learning after a supervised finetuning
  application: Knowledge-intensive dialog or language tasks
  has source code: https://github.com/openai/following-instructions-human-feedback
  blog post: https://sh-tsang.medium.com/review-instructgpt-training-language-models-to-follow-instructions-with-human-feedback-7fce4bf9059a, https://openai.com/research/instruction-following
  license: Closed source, accessible through API
  research problem: Large Language Models (LLMs), transformer model
  ```

- DALL-E 2

  ```yaml
  Title: Hierarchical Text-Conditional Image Generation with CLIP Latents
  model family: GLIDE, CLIP
  date created: 2022-04-01
  organization: OpenAI
  pretraining architecture: Encoder/Decoder
  pretraining task: Caption prediction
  training corpus: Combination of the DALL-E and CLIP datasets
  number of parameters: 3.5B
  maximum number of parameters (in million): 3500
  extension: Combines CLIP encoder and Diffusion decoder similar to GLIDE
  application: Text to image
  blog post: https://openai.com/product/dall-e-2, https://labs.openai.com/
  license: Closed source, accessible through API
  research problem: Large Language Models (LLMs), transformer model
  ```

- ChatGPT

  ```yaml
  Title: Introducing ChatGPT
  model family: GPT
  date created: 2022-11-30
  organization: OpenAI
  innovation: trained using Reinforcement Learning from Human Feedback (RLHF) to obtain better model alignment, It can generate upto 3000 words (equivalent to 6 pages), Supports input context length of 2048 tokens
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  fine-tuning task: Step 3. RLHF using Proximal Policy Optimization, Step 2. Collect comparison data and train a reward model, Step 1. Supervized fine-tuning
  training corpus: Human written prompt and interaction dataset collected through the OpenAI API
  number of parameters: 175B
  maximum number of parameters (in million): 175000
  hardware information: trained on an Azure AI supercomputing infrastructure
  application: provide human-like conversational interactions and assist users in answering questions, generating text, providing recommendations, and engaging in natural language conversations.
  blog post: https://openai.com/blog/chatgpt
  license: Closed source, accessible through API
  research problem: transformer model, Large Language Models (LLMs)
  ```

- GPT-4

  ```yaml
  Title: GPT-4 Technical Report
  model family: GPT
  date created: 2023-03-14
  organization: OpenAI
  innovation: It can generate upto 24000 words (equivalent to 48 pages), Supports input context length between 8192 and 32,768 tokens depending on the model version
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  fine-tuning task: Reinforcement Learning from Human Feedback, Rule-Based Reward Model
  number of parameters: 170T
  maximum number of parameters (in million): 170000000
  extension: a large-scale, multimodal model which can accept image and text inputs and produce text outputs
  application: Creating highly realistic and contextually accurate human-like text generation
  blog post: https://openai.com/research/gpt-4
  license: Closed source, accessible through API
  research problem: transformer model, Large Language Models (LLMs)
  ```

#### Google

- BERT

  ```yaml
  Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
  model family: BERT
  date created: 2018-10-01
  organization: Google
  innovation: BERT's primary innovation in Language Model Learning is the "masked language model" (MLM) approach, inspired by the Cloze task. This method masks random tokens in a sentence and trains the model to predict them, enabling bidirectional context understanding.
  pretraining architecture: Encoder
  pretraining task: Masked Language Modeling
  fine-tuning task: Next Sentence Prediction
  training corpus: Toronto Book Corpus and Wikipedia (3.3B Tokens)
  optimizer: Adam optimizer
  tokenization: WordPiece
  number of parameters: Base = 110M, Large = 340M
  maximum number of parameters (in million): 340
  application: General Language Understanding and Question Answering. Many other language applications followed
  has source code: https://huggingface.co/docs/transformers/model_doc/bert
  blog post: https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb, https://www.philschmid.de/bert-text-classification-in-a-different-language
  license: Open, Apache 2.0
  research problem: Large Language Models (LLMs), transformer model
  ```

- ALBERT

  ```yaml
  Title: ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
  model family: BERT
  date created: 2019-09-01
  organization: Google
  innovation: The main innovation of the work is ALBERT, a language model that improves on existing large models like BERT by employing parameter reduction techniques, such as factorized embeddings and cross-layer parameter sharing. This allows ALBERT to achieve better performance and efficiency on natural language understanding tasks by training larger models with fewer parameters.
  pretraining architecture: Encoder
  pretraining task: Next Sentence Prediction, Masked Language Modeling
  training corpus: Same as BERT
  optimizer: LAMB optimizer
  tokenization: sentencepiece
  number of parameters: Base = 12M, Large = 18M, XLarge = 60M
  maximum number of parameters (in million): 60
  hardware used: Cloud TPUv3
  extension: Compressed version of BERT using parameter sharing, which is much more efficient given the same number of parameters
  application: Same as BERT
  has source code: https://github.com/google-research/albert, https://huggingface.co/docs/transformers/model_doc/albert
  blog post: https://ai.googleblog.com/2019/12/albert-lite-bert-for-self-supervised.html
  license: Open, Apache 2.0
  research problem: Large Language Models (LLMs), transformer model
  ```

- T5

  ```yaml
  Title: Exploring the limits of transfer learning with a unified text-to-text transformer
  model family: T5
  date created: 2019-10-01
  organization: Google
  innovation: The main innovation of Google's T5 language model is its "text-to-text" framework, where various tasks are formulated as converting input text to output text. This unified approach allows T5 to achieve state-of-the-art performance on diverse tasks without task-specific modifications, simplifying training and deployment. This innovation enhances efficiency and effectiveness in real-world applications of large language models.
  pretraining architecture: Encoder/Decoder
  pretraining task: Span Corruption
  fine-tuning task: finetuning on downstream tasks one at a time
  training corpus: Colossal Clean Crawled Corpus
  optimizer: AdaFactor
  tokenization: sentencepiece
  number of parameters: 60M, 220M, 770M, 3B, and 11B
  maximum number of parameters (in million): 11000
  hardware used: TPUv3
  hardware information: we use a combination of model and data parallelism and train models on “slices” of Cloud TPU Pods. TPU pods are are multi-rack ML  supercomputers that contain 1,024 TPU v3 chips connected via a high-speed 2D mesh interconnect with supporting CPU host machines.
  extension: Same as original Transformer with some additions such as relative positional embeddings like Transformer XL
  application: Diverse set of downstream tasks including machine translation, question answering, abstractive summarization, and text classification
  has source code: https://github.com/google-research/text-to-text-transfer-transformer, https://huggingface.co/docs/transformers/model_doc/t5
  blog post: https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html
  license: Apache 2.0
  research problem: Large Language Models (LLMs), transformer model
  ```

- Big Bird

  ```yaml
  Title: Big Bird: Transformers for Longer Sequences
  model family: BERT
  date created: 2020-07-01
  organization: Google
  innovation: BigBird introduces a sparse attention mechanism, allowing it to efficiently handle sequences up to 8 times longer than traditional models like BERT. It combines global, sliding window, and random attention patterns to capture both local and long-range dependencies. This innovation enables superior performance on various NLP tasks without sacrificing efficiency.
  pretraining architecture: Encoder
  pretraining task: Masked Language Modeling
  training corpus: Books, CC-News, Stories and Wikipedia
  tokenization: byte pair encoding
  number of parameters: Depends on the overall architecture
  extension: Big Bird can extend other architectures such as BERT, Pegasus, or RoBERTa by using a sparse attention mechanism that elminates the quadratic dependency thus making it more suitable for longer sequences
  application: Particularly well suited for longer sequences, not only in text but also e.g. in genomics
  has source code: https://github.com/google-research/bigbird, https://huggingface.co/docs/transformers/model_doc/big_bird
  blog post: https://ai.googleblog.com/2021/03/constructing-transformers-for-longer.html, https://huggingface.co/blog/big-bird
  license: Open, Apache 2.0
  research problem: Large Language Models (LLMs), transformer model
  ```

- ViT

  ```yaml
  Title: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
  model family: BERT
  date created: 2020-10-01
  organization: Google
  innovation: The Vision Transformer (ViT) applies Transformers, typically used in NLP, directly to image patches without image-specific biases. It excels when pre-trained on larger datasets, outperforming traditional convolutional models like ResNets. This approach challenges the dominance of convolutional architectures in computer vision, mirroring the Transformer's rise in NLP.
  pretraining architecture: Encoder
  pretraining task: image classification
  training corpus: From standard Imagenet to JFT-300M (large inhouse dataset)
  optimizer: Adam optimizer
  number of parameters: 86M(Base) to 632M (Huge)
  maximum number of parameters (in million): 632
  hardware used: Cloud TPUv3
  hardware information: the ViT-L/16 model pre-trained on the public ImageNet-21k dataset could be trained using a standard cloud TPUv3 with 8 cores in approximately 30 days.
  extension: Extension of BERT architecture to train on patches of images
  application: image classification
  has source code: https://github.com/google-research/vision_transformer, https://huggingface.co/docs/transformers/model_doc/vit
  blog post: https://www.v7labs.com/blog/vision-transformer-guide
  license: N/A
  research problem: Large Language Models (LLMs), transformer model
  ```

- Switch

  ```yaml
  Title: Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity
  model family: T5
  date created: 2021-01-01
  organization: Google
  innovation: The Switch Transformer introduces a sparsely-activated model approach, enhancing the Mixture of Experts (MoE) models by simplifying their routing algorithm and reducing computational costs. It enables training large models with lower precision formats like bfloat16 and achieves up to 7x faster pre-training speeds. This innovation pushes LLM boundaries, scaling up to trillion parameter models with significant efficiency gains.
  pretraining architecture: Encoder/Decoder
  pretraining task: denoising autoencoder
  training corpus: Colossal Clean Crawled Corpus
  number of parameters: 1T
  maximum number of parameters (in million): 1000000
  hardware used: TPUv3
  hardware information: All models are trained with the same amount of computation (32 cores) and on the same hardware (TPUv3).
  extension: Goal to increase parameter count while keeping FLOP operations constant by using efficient routing of MoE (Mixture of Experts)
  application: General language tasks (e.g. question answering)
  has source code: https://github.com/google-research/t5x, https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py
  blog post: https://www.alexanderthamm.com/en/blog/switch-transformer-upscaling-to-over-a-billion-parameters/
  license: Open, Apache 2.0
  research problem: Large Language Models (LLMs), transformer model
  ```

- GLaM

  ```yaml
  Title: GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
  model family: Transformer
  date created: 2021-12-01
  organization: Google
  innovation: GLaM introduces a sparsely activated mixture-of-experts architecture, allowing it to scale to 1.2 trillion parameters while consuming only 1/3 of GPT-3's training energy. Despite its size, it achieves superior performance on 29 NLP tasks and is more energy-efficient than dense models like GPT-3.
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  training corpus: 1.6T tokens including web pages filtered by Wikipedia and books for quality
  optimizer: AdaFactor
  tokenization: sentencepiece
  number of parameters: 1.2T across 64 experts, but only 96B get activated for inference
  maximum number of parameters (in million): 1200000
  hardware used: cloud TPU-v4
  hardware information: the GLaM (64B/64E) training after 600B tokens consumes 456 MWh, about 1/3 of the energy cost of 1287 MWh used by GPT-3. Moreover, to reach similar (and slightly exceeded) scores as GPT-3, we train using 1,024 TPU-v4 chips for 574 hours (with 280B tokens). This consumes 213 MWh or 1/6 of the GPT-3 energy cost.
  extension: GLaM introduces a Mixture of 64 Experts to increase parameter count and generalization properties in a somewhat standard decoder-only. Transformer architecture. Only two experts get activated at a time per token, which makes the model also more efficient in training and inference.
  application: General language modeling - tested across 29 NLP tasks
  blog post: https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html
  license: closed source
  research problem: Large Language Models (LLMs), transformer model
  ```

- LAMDA

  ```yaml
  Title: LaMDA: Language Models for Dialog Applications
  model family: LaMDA-PT
  date created: 2022-01-01
  organization: Google
  innovation: LaMDA is a specialized dialog model that emphasizes safety and factual grounding. The model's innovation lies in its fine-tuning with annotated data and its ability to consult external knowledge sources. This approach aims to produce more accurate and safer dialog responses compared to traditional LLMs.
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  fine-tuning task: based on multi-turn crowdsourced dialog datasets, LaMDA-PT is finetuned in a mix of generative tasks that generate response given contexts, and discriminative tasks that evaluate quality and safety of a response in context 
  training corpus: 1.56T words from public dialog data and other public web documents. Overall, it consists of 2.97B documents, 1.12B dialogs, and 13.39B dialog utterances, for a total of 1.56T words
  tokenization: sentencepiece
  number of parameters: 137B
  maximum number of parameters (in million): 137000
  hardware used: TPUv3
  hardware information: LaMDA was pretrained on 1024 TPU-v3 chips for a total of about 57.7 days, and 256K tokens per batch
  extension: LAMDA focuses on how to improve safety, quality, and groundeness using different fine-tuning strategies
  application: General language modeling, such as translation, summarization, question and answers
  has source code: https://github.com/conceptofmind/LaMDA-rlhf-pytorch
  blog post: https://ai.googleblog.com/2022/01/lamda-towards-safe-grounded-and-high.html, https://blog.google/technology/ai/lamda/
  license: closed source
  research problem: Large Language Models (LLMs), transformer model
  ```

- FLAN

  ```yaml
  Title: Finetuned language models are zero-shot learners
  model family: LaMDA-PT
  date created: 2022-02-08
  organization: Google
  innovation: The primary innovation of FLAN in the context of Large Language Models is instruction tuning, where models are finetuned on datasets described via natural language instructions. This method significantly enhances zero-shot learning abilities, with FLAN outperforming the 175B GPT-3 on numerous tasks. The approach emphasizes human-like prompts over traditional model-specific prompts used in models like GPT-3 and T5.
  pretraining architecture: Decoder
  fine-tuning task: Instruction Tuning
  training corpus: FLAN is instruction tuned on 25 tasks spanning 62 datasets., LaMDA-PT is is pretrained on a collection of web documents (including those with computer code), dialog data, and Wikipedia, tokenized into 2.49T BPE tokens with a 32k vocabulary
  optimizer: AdaFactor
  tokenization: sentencepiece
  number of parameters: 137B
  maximum number of parameters (in million): 137000
  hardware used: TPUv3
  hardware information: instruction tuning takes around 60 hours on a TPUv3 with 128 cores
  extension: Zero-shot task learning. The output space for a given task is either one of several classes (classification) or free text (generation).
  application: language understanding and generation tasks such as inference, sentiment analysis, paraphrase, closed-book QA, reading comprehension, coreference, summarization, translation, commonsense reasoning, and struct-to-text
  has source code: https://github.com/google-research/FLAN
  blog post: http://rylanschaeffer.github.io/blog_posts/2022-01-20-google-brain-flan.html, https://ai.googleblog.com/2021/10/introducing-flan-more-generalizable.html
  license: Apache 2.0
  research problem: Large Language Models (LLMs), transformer model
  ```

- PaLM

  ```yaml
  Title: PaLM: Scaling Language Modeling with Pathways
  model family: PaLM
  date created: 2022-04-01
  organization: Google
  innovation: To demonstrate the first large-scale use of Pathways -- a new ML system which enables training a single model across thousands or tens of thousands of accelerator chips in a highly efficient manner. With Pathways, they trained a 540B parameter language model on 6144 TPU v4 chips at efficiency levels that could not be reached before for models of this scale. E.g., GPT-3 (175B), Gopher (280B), Megatron-Turing-NLG (530B).
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  training corpus: 780B tokens from multilingual social media conversations (50%), multilingual filtered webpages (27%), books in English (13%), code from Github (5%), multilingual Wikipedia (4%), and news in English (1%). Code includes 24 programming languages.
  optimizer: AdaFactor
  tokenization: sentencepiece
  number of parameters: 8B, 62B, and 540B
  maximum number of parameters (in million): 540000
  hardware used: TPUv4
  hardware information: PaLM 540B is trained over two TPU v4 Pods connected over data center network (DCN) using a combination of model and data parallelism. Each Pod has 3072 TPU v4 chips attached to 768 hosts.
  extension: PaLM uses a typical decoder-only transformer architecture, but adds quite a few extensions: SwiGLU activations, parallel layers, multi-query attention, RoPE embeddings, Shared Input-Output Embeddings, no biases, and a 256k SentencePiece vocabulary generated from the training data
  application: PaLM is designed as a general purpose language model with applicability to hundreds of different language tasks
  has source code: https://github.com/lucidrains/PaLM-pytorch
  blog post: https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/, https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html
  license: closed source
  research problem: Large Language Models (LLMs), transformer model
  ```

- UL2

  ```yaml
  Title: Ul2: Unifying language learning paradigms
  model family: Transformer
  date created: 2022-05-01
  organization: Google
  innovation: The paper introduces the UL2 model, a unified framework for pre-training in NLP, featuring a novel Mixture-of-Denoisers (MoD) objective. This objective smoothly integrates various pre-training paradigms, such as span corruption and prefix language modeling. Additionally, UL2 introduces dynamic "mode switching" between different denoisers and showcases superior performance across diverse NLP tasks.
  pretraining architecture: Encoder/Decoder
  pretraining task: Mixture-of-Denoisers, which combines diverse pretraining paradigms together
  training corpus: 1 trillion tokens on C4
  optimizer: AdaFactor
  tokenization: sentencepiece
  number of parameters: 20B
  maximum number of parameters (in million): 20000
  hardware used: TPUv4
  hardware information: We use a batch size of 1024 and 512 TPUv4 chips for pretraining this model. UL20B is trained with Jax and T5X infrastructure. We release and open source T5X-based model checkpoints of this 20B model
  extension: UL2-20B (Unifying Language Learning) can be interpreted as a model that is quite similar to T5 but trained with a different objective and slightly different scaling knobs.
  application: A unified framework for pre-training models that are universally effective across datasets and setups.
  has source code: https://github.com/google-research/google-research/tree/master/ul2
  blog post: https://blog.research.google/2022/10/ul2-20b-open-source-unified-language.html
  license: Open, Apache 2.0
  research problem: Large Language Models (LLMs), transformer model
  ```

- Imagen

  ```yaml
  Title: Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding
  model family: Diffusion models, CLIP, T5
  date created: 2022-06-01
  organization: Google
  innovation: The "Imagen" model innovatively merges transformer language models with high-fidelity diffusion techniques to produce photorealistic images from text descriptions. This demonstrates that embeddings from text-only pretrained large language models are highly effective for text-to-image synthesis.
  pretraining architecture: T5 (or CLIP or BERT) for frozen text encoder + U-net architecture for cascaded diffusion models for text to image
  pretraining task: image/text pair prediction
  training corpus: a combination of internal datasets, with ? 460M image-text pairs, and the publicly available Laion dataset, with ? 400M image-text pairs
  optimizer: AdaFactor
  number of parameters: 2B
  maximum number of parameters (in million): 2000
  hardware used: TPUv4
  hardware information: use 256 TPU-v4 chips for our base 64 x 64 model, and 128 TPU-v4 chips for both super-resolution models
  extension: Imagen adds a few extensions to the U-net diffusion architecture (pooled embedding vector, cross attention over text embeddings, and Layer Normalizations)
  application: Text to image
  blog post: https://imagen.research.google/
  license: closed source
  research problem: Large Language Models (LLMs), transformer model
  ```

- Minerva

  ```yaml
  Title: Solving Quantitative Reasoning Problems with Language Models
  model family: PaLM
  date created: 2022-06-01
  organization: Google
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  training corpus: Same as PaLM + 118GB dataset of scientific papers from the arXiv preprint server and web pages that contain mathematical expressions using LaTeX, MathJax, or other mathematical typesetting formats
  number of parameters: 540B
  maximum number of parameters (in million): 540000
  extension: Extends PaLM by fine-tuning on the mathematical dataset
  application: Mathematical reasoning
  blog post: https://ai.googleblog.com/2022/06/minerva-solving-quantitative-reasoning.html
  license: closed source
  research problem: Large Language Models (LLMs), transformer model
  ```

- Flan-T5

  ```yaml
  Title: Scaling instruction-finetuned language models
  model family: T5
  date created: 2022-11-01
  organization: Google
  innovation: this paper explores instruction finetuning with a particular focus on (1) scaling the number of tasks (1.8K fine-tuning tasks), (2) scaling the model size, and (3) finetuning on chain-of-thought data. This approach is compatible with various model sizes and architectures, with Flan-T5 models notably outperforming baseline T5 models.
  pretraining architecture: Encoder/Decoder
  pretraining task: Span Corruption
  fine-tuning task: Instruction Tuning
  training corpus: Flan finetuned with tasks in Muffin, T0-SF, NIV2, and CoT
  optimizer: AdaFactor
  number of parameters: 80M (Flan-T5-Small), 250M (Flan-T5-Base), 780M (FLan-T5-Large), 3B (Flan-T5-XL), and 11B (Flan-T5-XXL).
  maximum number of parameters (in million): 11000
  hardware used: TPUv3
  extension: instruction finetuning with a particular focus on (1) scaling the number of tasks, (2) scaling the model size, and (3) finetuning on chain-of-thought data
  application: The primary use is to underestand how to improve large language models with the right kind of instruction fine-tuning. The focus is research on zero-shot and in-context few-shot learning NLP tasks, such as reasoning, and question answering; advancing fairness and safety research, and understanding limitations of current large language models
  has source code: https://github.com/google-research/t5x, https://huggingface.co/docs/transformers/model_doc/flan-t5
  blog post: https://ai.googleblog.com/2023/02/the-flan-collection-advancing-open.html
  license: Apache 2.0
  research problem: Large Language Models (LLMs), transformer model
  ```

- Flan-PaLM

  ```yaml
  Title: Scaling instruction-finetuned language models
  model family: PaLM
  date created: 2022-11-01
  organization: Google
  innovation: The paper introduced an extended instruction fine-tuning for the Flan-PaLM model, scaling it to a 540B-parameter size and 1.8K fine-tuning tasks. They incorporated chain-of-thought (CoT) data, which enhanced performance across evaluations. This approach is compatible with various model sizes and architectures
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  fine-tuning task: Instruction Tuning
  training corpus: Flan finetuned with tasks in Muffin, T0-SF, NIV2, and CoT
  optimizer: AdaFactor
  number of parameters: 8B, 62B, 540B
  maximum number of parameters (in million): 540000
  hardware used: TPUv4
  hardware information: use 0.2% of the pre-training compute to instruction-finetune Flan-PaLM 540B (approximately 512 v4 TPU chips for 37 hours)
  extension: Flan-PaLM is generated by "Flan Finetuning" the PaLM models: (1) scaling the number of tasks to 1,836, (2) scaling the model size, and (3) finetuning on chain-of-thought data.
  application: Same as Flan-T5. The goal is to show Flan finetuning can even improve on the largest Google LMs (+9.4% improvement average across tasks), with improvements to chain of thought, self consistency, multilingual tasks, arithmetic reasoning
  license: closed source
  research problem: Large Language Models (LLMs), transformer model
  ```

#### Google, and CMU

- Transformer XL

  ```yaml
  Title: Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context
  date created: 2019-01-01
  organization: Google, CMU
  innovation: Transformer-XL introduces a segment-level recurrence mechanism and a novel positional encoding scheme to overcome the fixed-length context limitations of traditional Transformers. This allows it to capture dependencies 80% longer than RNNs and 450% longer than vanilla Transformers, addressing context fragmentation and improving efficiency in language modeling.
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  training corpus: Different training datasets depending on experiments, but baseline is Wikitext-103
  tokenization: byte pair encoding
  number of parameters: 151M
  maximum number of parameters (in million): 151
  hardware information: state-of-the-art results reported in the paper were obtained by training the model on a large-scale TPU cluster
  extension: Relative positioned embeddings enable longer-context attention when compared to vanilla Transformer model
  application: General language tasks
  has source code: https://github.com/chiayewken/transformer_xl, https://huggingface.co/docs/transformers/model_doc/transfo-xl
  blog post: https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html
  license: N/A
  research problem: Large Language Models (LLMs), transformer model
  ```

- XLNet

  ```yaml
  Title: XLNet: Generalized Autoregressive Pretraining for Language Understanding
  model family: Transformer XL
  date created: 2019-05-01
  organization: Google, CMU
  innovation: XLNet introduces a generalized autoregressive pretraining method that captures bidirectional context by considering all possible permutations of the factorization order. This approach overcomes BERT's limitations related to data corruption and token independence. Additionally, XLNet integrates techniques from Transformer-XL and offers architectural improvements for permutation-based modeling.
  pretraining architecture: Decoder
  pretraining task: Causal language modeling
  training corpus: Same as BERT + Giga5 (16GB text), and and aggressively filtered ClueWeb 2012-B (19GB), Common Crawl (110 GB)
  optimizer: Adam weight decay optimizer
  number of parameters: Base=117M, Large=360M
  maximum number of parameters (in million): 360
  hardware used: TPUv3
  hardware information: train on 512 TPU v3 chips for 500K steps with an Adam weight decay optimizer, linear learning rate decay, and a batch size of 8192, which takes about 5.5 days
  extension: This model basically adapts Transformer XL architecture to permutation-based LM
  application: General language tasks
  has source code: https://huggingface.co/docs/transformers/model_doc/xlnet
  blog post: https://towardsdatascience.com/xlnet-explained-in-simple-terms-255b9fb2c97c
  license: Open, MIT license
  research problem: Large Language Models (LLMs), transformer model
  ```
  
#### Pengcheng Lab, and Baidu

- ERNIE

  ```yaml
  Title: ERNIE: Enhanced Language Representation with Informative Entities
  model family: BERT
  date created: 2019-05-01
  organization: Pengcheng Lab, Baidu
  innovation: ERNIE innovatively incorporates knowledge from knowledge graphs (KGs) into language representation models. It fuses lexical, syntactic, and knowledge information, enabling enhanced performance on knowledge-driven tasks. This approach sets ERNIE apart from traditional models like BERT, which primarily rely on textual context.
  pretraining architecture: Encoder
  pretraining task: Masked Language Modeling
  training corpus: English Wikipedia + Wikidata for entitites (note that they initialize model to original BERT parameter values
  optimizer: Adam optimizer
  number of parameters: Ernie-ViLG 2.0 = 10B, Ernie 3.0 Titan = 260B
  maximum number of parameters (in million): 260000
  extension: Uses BERT for Encoder architecture, but stacks and aggregates two of them for text and entities. This architecture could be understood as BERT for text + knowledge graphs
  application: Knowledge intensive related tasks that might benefit from knowledge graphs or entities such as entity recognition
  has source code: https://github.com/thunlp/ERNIE
  blog post: http://research.baidu.com/Blog/index-view?id=160
  license: closed source
  research problem: Large Language Models (LLMs), transformer model
  ```



## License

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
