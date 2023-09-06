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

  ```yaml
  Title: Improving Language Understanding by Generative Pre-Training
  has model: GPT-1
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

  ```yaml
  Title: Language models are unsupervised multitask learners
  has model: GPT-2
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

  ```yaml
  Title: Language Models are Few-Shot Learners
  has model: GPT-3
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

  ```yaml
  Title: Zero-Shot Text-to-Image Generation
  has model: DALL-E
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

  ```yaml
  Title: Learning Transferable Visual Models From Natural Language Supervision
  has model: CLIP
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

  ```yaml
  Title: GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models
  has model: GLIDE
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

  ```yaml
  Title: Training language models to follow instructions with human feedback
  has model: InstructGPT
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

  ```yaml
  Title: Hierarchical Text-Conditional Image Generation with CLIP Latents
  has model: DALL-E 2
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

  ```yaml
  Title: Introducing ChatGPT
  has model: ChatGPT
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

  ```yaml
  Title: GPT-4 Technical Report
  has model: GPT-4
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

  ```yaml
  Field: Language
  Params: 40B
  Training Data: 1T tokens (RefinedWeb)
  License: Apache 2.0
  Context Length: 2048
  ```

#### Google, and CMU

  ```yaml
  Field: Language
  Params: 13B, 7B, 3B
  Training Data: 1T tokens (RedPajama)
  License: Apache 2.0
  Context Length: 2048
  ```

- **Redpajama-INCITE** [[Together]](https://github.com/togethercomputer/RedPajama-Data) May. 2023 [[open]](https://huggingface.co/togethercomputer/RedPajama-INCITE-Base-3B-v1)

  ```yaml
  Field: Language
  Params: 7B, 3B
  Training Data: 1T tokens (Redpajama)
  License: Apache 2.0
  Context Length: 2048
  ```

- **MPT** [[MosaicML]](https://www.mosaicml.com/blog/mpt-7b) May. 2023 [[open]](https://github.com/mosaicml/llm-foundry)  

  ```yaml
  Field: Language
  Params: 30B, 7B
  Training Data: 1T tokens (Private)
  License: Apache 2.0, CC BY-SA-3.0
  Context Length: 84k
  ```

- **Stable-LM** [[Stability-AI]](https://stability.ai/blog/stability-ai-launches-the-first-of-its-stablelm-suite-of-language-models) Apr. 2023 [[open]](https://github.com/Stability-AI/StableLM#stablelm-alpha)

  ```yaml
  Field: Language
  Params: 7B, 3B
  Training Data: 1.5T tokens
  License: CC BY-SA-4.0
  ```

- **LiT-LLaMa** [[Lightning-AI]]() Apr. 2023 [[open]](https://github.com/Lightning-AI/lit-llama)  
  
  ```yaml
  Field: Language
  Params: 13B, 7B
  Training Data: 1.2T tokens (Redpajama)
  License: Apache 2.0
  ```

- **h2oGPT** [[H2O.ai]](https://h2o.ai/blog/building-the-worlds-best-open-source-large-language-model-h2o-ais-journey/) [[open]](https://github.com/h2oai/h2ogpt)  
  [h2oGPT: Democratizing Large Language Models](https://arxiv.org/pdf/2306.08161.pdf)

  ```yaml
  Field: Language
  Params: 13B, 7B
  Training Data: 1.0T tokens
  License: Apache 2.0
  Context Length: 2048
  ```

- **Cerabras-GPT** [[Cerabras]]() Mar. 2023 [[open]](https://huggingface.co/cerebras/Cerebras-GPT-13B)  
  Training Compute-Optimal Large Language Models [[preprint]](https://arxiv.org/abs/2203.15556)  

  ```yaml
  Field: Language
  Params: 13B
  Training Data: 371B tokens (Redpajama)
  License: Apache 2.0
  Context Length: 2048
  ```

- **Claude** [[Anthropic]](https://www.anthropic.com/index/introducing-claude) Mar. 2023 [close]

  ```yaml
  Field: Language-Vision
  ```

- **GPT-4** [[OpenAI]](https://openai.com/product/gpt-4) Mar. 2023 [close]  
   GPT-4 Technical Report [[Preprint]](https://cdn.openai.com/papers/gpt-4.pdf)

  ```yaml
  Field: Language-Vision
  Params: 1.7T
  Architecture: De, MoE
  ```

- **Bard** [[Google]](https://blog.google/technology/ai/bard-google-ai-search-updates/)
  
  ```yaml
  Field: Language-Vision
  ```

- **LLaMa** [[Meta]]() Feb. 2023 [[open]](https://github.com/facebookresearch/llama)  
   Open and Efficient Foundation Language Models [[Preprint]](https://arxiv.org/pdf/2302.13971v1.pdf)

  ```yaml
  Field: Language
  Params: 65B, 33B, 13B, 7B
  Training Data: 4TB (1.4T tokens)
  Training Cost: 1,022,362 (2048 80G-A100 x 21 days)
  Training Power Consumption: 449 MWh
  Instruction-tuned Variants: Alpaca, Vicuna, Dolly, Guanaco, ColossalChat, GPT4All, Koala, BELLE, MiniGPT-4, etc.
  License: GPL
  ```

- **RWKV-4** [[Personal]]() Dec. 2022 [[open]](https://github.com/BlinkDL/RWKV-LM)

  ```yaml
  Field: Language
  Params: 14B, 7B, 3B, 1.5B
  Training Data: 332B tokens
  Architecture: De, RNN
  License: Apache 2.0
  ```

- **AnthropicLM** [[Anthropic]]() Dec. 2022 [close]  
   Constitutional AI: Harmlessness from AI Feedback

  ```yaml
  Field: Language
  Params: 52B
  ```

- **BLOOM** [[BigScience]]() Nov. 2022 [[open]](https://huggingface.co/bigscience/bloom)  
   A 176B-Parameter Open-Access Multilingual Language Model [[Preprint]](https://arxiv.org/pdf/2211.05100.pdf)

  ```yaml
  Field: Language
  Params: 176B
  Training Data: 174GB (336B tokens)
  Training Cost: 1M A100 GPU hours = 384 80G-A100 x 4 months
  Training Power Consumption: 475 MWh
  Training Framework: Megatron + Deepspeed
  Instruction-tuned Variants: BLOOMZ
  License: OpenRAIL-M v1
  Context Length: 2048
  ```
  
- **Galactica** [[Meta]]() Nov. 2022 [[open]](https://huggingface.co/facebook/galactica-1.3b)
  A scientific language model trained on over 48 million scientific texts [[Preprint]](https://arxiv.org/pdf/2211.09085.pdf)
  
  ```yaml
  Field: Language
  Params: 125M, 1.3B, 6.7B, 30B, 120B
  ```

- **Pythia** [[EleutherAI]]() Oct. 2022 [[open]](https://github.com/EleutherAI/pythia)
  
  ```yaml
  Field: Language
  Params: 12B
  Instruction-tuned Variants: Dolly 2.0
  License: Apache 2.0
  Context Length: 2048
  ```

- **GLM-130B** [[BAAI]](https://keg.cs.tsinghua.edu.cn/glm-130b/zh/posts/glm-130b/) Oct. 2022 [[open]](https://github.com/THUDM/GLM-130B)  
   GLM-130B: An Open Bilingual Pre-trained Model [[ICLR'23]](https://arxiv.org/pdf/2210.02414.pdf)

  ```yaml
  Field: Language
  Params: 130B
  Training Data: (400B tokens)
  Training Cost: 516,096 A100 hours = 768 40G-A100 x 28 days
  Training Framework: Megatron + Deepspeed
  ```

- **UL2** [[Google]]() May 2022 [[open]](https://huggingface.co/google/ul2)  
   Unifying Language Learning Paradigms [[Preprint]](https://arxiv.org/abs/2205.05131)

  ```yaml
  Field: Language
  Params: 20B (1T tokens)
  Training Data: 800GB
  Achitecture: En-De
  Training Framework: Jax + T5x
  License: Apache 2.0
  Instruction-tuned Variants: Flan-UL2
  Context Length: 2048
  ```

- **OPT** [[Meta]](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/) May 2022 [[open]](https://github.com/facebookresearch/metaseq)  
   OPT: Open Pre-trained Transformer Language Models [[Preprint]](https://arxiv.org/abs/2205.01068)

  ```yaml
  Field: Language
  Params: 175B
  Training Data: 800GB (180B tokens)
  Training Cost: 809,472 A100 hours =  992 80G-A100 x 34 days
  Training Power Consumption: 356 MWh
  Architecutre: De
  Training Framework: Megatron + Fairscale
  ```

- **PaLM** [[Google]](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html) Apr. 2022 [close]  
   PaLM: Scaling Language Modeling with Pathways [[Preprint]](https://arxiv.org/abs/2204.02311)

  ```yaml
  Field: Language
  Params: 550B
  Training Data: 3TB (780B tokens)
  Training Cost: $10M (16,809,984 TPUv4core-hours, 64 days)
  Training petaFLOPs: 2.5B
  Architecture: De
  Training Framework: Jax + T5x
  ```

- **GPT-NeoX** [[EleutherAI]](https://blog.eleuther.ai/announcing-20b/) Apr. 2022 [[open]](https://github.com/EleutherAI/gpt-neox)  
   GPT-NeoX-20B: An Open-Source Autoregressive Language Model [[Preprint]](https://arxiv.org/abs/2204.06745)

  ```yaml
  Field: Language
  Params: 20B
  Training Data: 525GiB
  Training petaFLOPs: 93B
  Architecture: De
  Training Framework: Megatron + Fairscale
  License: Apache 2.0
  Context Length: 2048
  ```

- **InstructGPT** [[OpenAI]]() Mar. 2022 [close]  
   Training language models to follow instructions with human feedback [[Preprint]](https://arxiv.org/abs/2203.02155)

  ```yaml
  Field: Language
  Params: 175B
  ```

- **Chinchilla** [[DeepMind]](https://www.deepmind.com/publications/an-empirical-analysis-of-compute-optimal-large-language-model-training) Mar. 2022 [close]  
   Training Compute-Optimal Large Language Models [[Preprint]](https://arxiv.org/abs/2203.15556)

  ```yaml
  Field: Language
  Params: 70B
  Training Data: 5.2TB (1.4T tokens)
  Training petaFLOPs: 580M
  Architecture: De
  ```

- **EVA 2.0** [[BAAI]](https://wudaoai.cn/model/detail/EVA) Mar. 2022 [[open]](https://openi.pcl.ac.cn/BAAI/WuDao-Model/src/branch/master)  
   EVA2.0: Investigating Open-Domain Chinese Dialogue Systems with Large-Scale Pre-Training [[Preprint]](https://arxiv.org/abs/2203.09313)

  ```yaml
  Field: Language (Dialogue)
  Params: 2.8B
  Training Data: 180G (1.4B samples, Chinese)
  ```

- **AlphaCode** [[DeepMind]](https://www.deepmind.com/blog/competitive-programming-with-alphacode) Mar. 2022 [close]  
   Competition-Level Code Generation with AlphaCode [[Preprint]](https://arxiv.org/abs/2203.07814)

  ```yaml
  Field: Code Generation
  Params: 41B
  Training Data: (967B tokens)
  Architecture: De
  ```

- **ST-MoE** [[Google]]() Feb. 2022 [close]  
   ST-MoE: Designing Stable and Transferable Sparse Expert Models [[Preprint]](https://arxiv.org/abs/2202.08906)

  ```yaml
  Field: Language
  Params: 296B
  Architecture: En-De, MoE
  ```

- **LaMDA** [[Google]](https://arxiv.org/abs/2201.08239) Jan. 2022 [close]  
   LaMDA: Language Models for Dialog Applications [[Preprint]](https://arxiv.org/abs/2201.08239)

  ```yaml
  Field: Language (Dialogue)
  Params: 137B
  Training Data: (1.56T words)
  Training petaFLOPs: 360M
  Architecture: De
  ```

- **GLaM** [[Google]](https://ai.googleblog.com/2021/12/more-efficient-in-context-learning-with.html) Dec. 2021 [close]  
   GLaM: Efficient Scaling of Language Models with Mixture-of-Experts [[Preprint]](https://arxiv.org/abs/2112.06905)

  ```yaml
  Field: Language
  Params: 1.2T
  Architecture: De, MoE
  ```

- **Gopher** [[DeepMind]](https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval) Dec. 2021 [close]  
   Scaling Language Models: Methods, Analysis & Insights from Training Gopher [[Preprint]](https://arxiv.org/abs/2112.11446)

  ```yaml
  Field: Language
  Params: 280B
  Training Data: 1.3TB (300B tokens)
  Training petaFLOPs: 630M
  Architecture: De
  ```

- **Yuan 1.0** [[inspur]](https://air.inspur.com/home) Oct. 2021 [close]  
   Yuan 1.0: Large-Scale Pre-trained Language Model in Zero-Shot and Few-Shot Learning [[Preprint]](https://arxiv.org/abs/2110.04725)

  ```yaml
  Field: Language
  Params: 245B
  Training Data: 5TB (180B tokens, Chinese)
  Training petaFLOPs: 410M
  Architecture: De, MoE
  ```

- **MT-NLG** [[Microsoft, Nvidia]](https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/) Oct. 2021 [close]  
   Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model [[Preprint]](https://arxiv.org/abs/2201.11990)

  ```yaml
  Field: Language
  Params: 530B
  Training Data: 339B tokens
  Training petaFLOPs: 1.4B
  Architecture: De
  ```

- **Plato-XL** [[Baidu]](http://research.baidu.com/Blog/index-view?id=163) Sept. 2021 [close]  
   PLATO-XL: Exploring the Large-scale Pre-training of Dialogue Generation [[Preprint]](https://arxiv.org/abs/2109.09519)

  ```yaml
  Field: Language (Dialogue)
  Params: 11B
  Training Data: (1.2B samples)
  ```

- **GPT-J** [[EleutherAI]](https://arankomatsuzaki.wordpress.com/2021/06/04/gpt-j/) Aug. 2021 [[open]](https://github.com/kingoflolz/mesh-transformer-jax)  

  ```yaml
  Field: Language
  Params: 6B
  Programming Language: Jax
  ```

- **Jurassic-1** [[AI21 Labs]](https://www.zdnet.com/article/watch-out-gpt-3-here-comes-ai21s-jurassic-language-model/) Aug. 2021 [close]  
   Jurassic-1: Technical Details and Evaluation [[Preprint]](https://uploads-ssl.webflow.com/60fd4503684b466578c0d307/61138924626a6981ee09caf6_jurassic_tech_paper.pdf)

  ```yaml
  Field: Language
  Params: 178B
  Training petaFLOPs: 370M
  Architecture: De
  ```

- **Codex** [[OpenAI]](https://openai.com/blog/openai-codex/) July 2021 [close]  
   Evaluating Large Language Models Trained on Code [[Preprint]](https://arxiv.org/abs/2107.03374)

  ```yaml
  Field: Code Generation
  Params: 12B
  Training Data: 159GB
  Architecture: De
  ```

- **ERNIE 3.0** [[Baidu]](https://wenxin.baidu.com/wenxin/ernie) July 2021 [close]  
   ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation [[Preprint]](https://arxiv.org/abs/2107.02137)

  ```yaml
  Field: Language
  Params: 10B
  Training Data: 4TB (375B tokens, with knowledge graph)
  Architecture: En
  Objective: MLM
  ```

- **CPM-2** [[BAAI]]() June 2021 [[open]](https://openi.pcl.ac.cn/BAAI/WuDao-Model/src/branch/master)  
   CPM-2: Large-scale Cost-effective Pre-trained Language Models [[Preprint]](https://arxiv.org/abs/2106.10715)

  ```yaml
  Field: Language
  Params: 198B
  Training Data: 2.6TB (Chinese 2.3TB, English 300GB)
  Architecture: En-De
  Objective: MLM
  ```

- **HyperClova** [[Naver]](https://www.navercorp.com/promotion/pressReleasesView/30546) May 2021 [close]  
   What Changes Can Large-scale Language Models Bring? Intensive Study on HyperCLOVA: Billions-scale Korean Generative Pretrained Transformers [[Preprint]](https://arxiv.org/abs/2109.04650v1)

  ```yaml
  Field: Language
  Params: 82B
  Training Data: 562B tokens (Korean)
  Training petaFLOPs: 63B
  Architecture: De
  ```

- **ByT5** [[Google]]() May 2021 [[open]](https://github.com/google-research/byt5)  
   ByT5: Towards a token-free future with pre-trained byte-to-byte models [[TACL'22]](https://arxiv.org/abs/2105.13626)

  ```yaml
  Field: Language
  Params: 13B
  Training Data: (101 languages)
  Architecture: En-De
  ```

- **PanGu-α** [[Huawei]]() Apr. 2021 [close]  
   PanGu-α: Large-scale Autoregressive Pretrained Chinese Language Models with Auto-parallel Computation [[Preprint]](https://arxiv.org/abs/2104.12369)

  ```yaml
  Field: Language
  Params: 200B
  Training Data: 1.1TB (Chinese)
  Training petaFLOPs: 58M
  Architecture: De
  ```

- **mT5** [[Google]]() Mar. 2021 [[open]](https://github.com/google-research/multilingual-t5)  
   mT5: A massively multilingual pre-trained text-to-text transformer [[Preprint]](https://arxiv.org/abs/2010.11934)

  ```yaml
  Field: Language
  Params: 13B
  Training Data: (101 languages)
  Architecture: En-De
  ```

- **WuDao-WenHui** [[BAAI]]() Mar. 2021 [[open]](https://openi.pcl.ac.cn/BAAI/WuDao-Model/src/branch/master/Transformer-XL)

  ```yaml
  Field: Language
  Params: 2.9B
  Training Data: 303GB (Chinese)
  ```

- **GLM** [[BAAI]]() Mar. 2021 [[open]](https://openi.pcl.ac.cn/BAAI/WuDao-Model/src/branch/master/GLM)  
   GLM: General Language Model Pretraining with Autoregressive Blank Infilling [[Preprint]](https://arxiv.org/abs/2103.10360)

  ```yaml
  Field: Language
  Params: 10B
  Architecture: De
  ```

- **Switch Transformer** [[Google]]() Jan. 2021 [[open]](https://github.com/google-research/t5x)  
   Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity [[Preprint]](https://arxiv.org/abs/2101.03961)

  ```yaml
  Field: Language
  Params: 1.6T
  Training Data: 750GB
  Training petaFLOPs: 82M
  Architecture: En-De, MoE
  Objective: MLM
  ```

- **CPM** [[BAAI]]() Dec. 2020 [[open]](https://github.com/TsinghuaAI/CPM)  
   CPM: A Large-scale Generative Chinese Pre-trained Language Model [[Preprint]](https://arxiv.org/abs/2012.00413)

  ```yaml
  Field: Language
  Params: 2.6B
  Training Data: 100G (Chinese)
  Training petaFLOPs: 1.8M
  Architecture: De
  Objective: LTR
  ```

- **GPT-3** [[OpenAI]](https://openai.com/api/) May 2020 [close]  
   Language Models are Few-Shot Learners [[NeurIPS'20]](https://papers.nips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf)

  ```yaml
  Field: Language
  Params: 175B
  Training Data: 45TB (680B Tokens)
  Training Time: 95 A100 GPU years (835584 A100 GPU hours, 355 V100 GPU years)
  Training Cost: $4.6M
  Training petaFLOPs: 310M
  Architecture: De
  Obective: LTR
  Instruction-tuned Variants: InstructGPT, WebGPT, ChatGPT
  ```

- **Blender** [[Meta]](https://ai.facebook.com/blog/blender-bot-2-an-open-source-chatbot-that-builds-long-term-memory-and-searches-the-internet/) Apr. 2020 [[close]](https://huggingface.co/facebook/blenderbot-90M?text=Hey+my+name+is+Thomas%21+How+are+you%3F)  
   Recipes for building an open-domain chatbot [[Preprint]](https://arxiv.org/abs/2004.13637)

  ```yaml
  Field: Language (Dialogue)
  Params: 9.4B
  ```

- **T-NLG** [[Microsoft]](https://www.microsoft.com/en-us/research/blog/turing-nlg-a-17-billion-parameter-language-model-by-microsoft/) Feb. 2020 [close]

  ```yaml
  Field: Language
  Params: 17B
  Training petaFLOPs: 16M
  Architecture: De
  Obective: LTR
  ```

- **Meena** [[Google]](https://ai.googleblog.com/2020/01/towards-conversational-agent-that-can.html) Jan. 2020 [close]  
   Towards a Human-like Open-Domain Chatbot [[Preprint]](https://arxiv.org/abs/2001.09977)

  ```yaml
  Field: Language (Dialogue)
  Params: 2.6B
  Training Data: 341GB (40B words)
  Training petaFLOPs: 110M
  ```

- **DialoGPT** [[Microsoft]](https://www.microsoft.com/en-us/research/project/large-scale-pretraining-for-response-generation/) Nov. 2019 [[open]](https://github.com/microsoft/DialoGPT)  
   DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation [[ACL'20]](https://arxiv.org/abs/1911.00536)

  ```yaml
  Field: Language (Dialogue)
  Params: 762M
  Training Data: (147M conversation)
  Architecture: De
  ```

- **T5** [[Google]](https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html) Oct. 2019 [[open]](https://github.com/google-research/text-to-text-transfer-transformer)  
   Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer [[JMLR'19]](https://arxiv.org/abs/1910.10683)

  ```yaml
  Field: Language
  Params: 11B
  Training Data: 800GB
  Training Cost: $1.5M
  Training petaFLOPs: 41M
  Architecture: En-De
  Obective: MLM
  License: Apache 2.0
  Instruction-tuned Variants: Flan-T5
  Context-Length: 512
  ```

- **Megatron-LM** [[Nvidia]]() Sept. 2019 [[open]](https://github.com/NVIDIA/Megatron-LM)  
   Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism [[Preprint]](https://arxiv.org/abs/1909.08053)

  ```yaml
  Field: Language
  Params: 8.3B
  Training Data: 174GB
  Training petaFLOPs: 9.1M
  Architecture: De
  Obective: LTR
  Training Framework: Megatron
  ```

- **Megatron-BERT** [[Nvidia]]() Sept. 2019 [[open]](https://github.com/NVIDIA/Megatron-LM)  
   Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism [[Preprint]](https://arxiv.org/abs/1909.08053)

  ```yaml
  Field: Language
  Params: 3.9B
  Training Data: 174GB
  Training petaFLOPs: 57M
  Architecture: En
  Obective: MLM
  Training Framework: Megatron
  ```

- **RoBERTa** [[Meta]](https://ai.facebook.com/blog/roberta-an-optimized-method-for-pretraining-self-supervised-nlp-systems/) July 2019 [[open]](https://github.com/facebookresearch/fairseq)  
   RoBERTa: A Robustly Optimized BERT Pretraining Approach [[Preprint]](https://arxiv.org/abs/1907.11692)

  ```yaml
  Field: Language
  Params: 354M
  Training Data: 160GB
  Training Time: 1024 V100 GPU days
  Architecture: En
  Objective: MLM
  ```

- **XLNet** [[Google]]() June 2019 [[open]](https://github.com/zihangdai/xlnet)  
   XLNet: Generalized Autoregressive Pretraining for Language Understanding [[NeurIPS'19]](https://papers.nips.cc/paper/2019/hash/dc6a7e655d7e5840e66733e9ee67cc69-Abstract.html)

  ```yaml
  Field: Language
  Params: 340M
  Training Data: 113GB (33B words)
  Training Time: 1280 TPUv3 days
  Training Cost: $245k
  Architecture: En
  Objective: PLM
  ```

- **GPT-2** [[OpenAI]](https://openai.com/blog/better-language-models/) Feb. 2019 [[open]](https://github.com/openai/gpt-2)  
   Language Models are Unsupervised Multitask Learners [[Preprint]](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

  ```yaml
  Field: Language
  Params: 1.5B
  Training Data: 40GB (8M web pages)
  Training Cost: $43k
  Training petaFLOPs: 1.5M
  Architecture: De
  Objective: LTR
  ```

- **BERT** [[Google]]() Oct. 2018 [[open]](https://github.com/google-research/bert)  
   BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [[NAACL'18]](https://arxiv.org/abs/1810.04805)

  ```yaml
  Field: Language
  Params: 330M
  Training Data: 16GB (3.3B words)
  Training Time: 64 TPUv2 days (280 V100 GPU days)
  Training Cost: $7k
  Training petaFLOPs: 290k
  Architecture: En
  Objective: MLM, NSP
  ```

- **GPT** [[OpenAI]](https://openai.com/blog/language-unsupervised/) June 2018 [open]
  Improving Language Understanding by Generative Pre-Training [[Preprint]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

  ```yaml
  Field: Language
  Params: 117M
  Training Data: 1GB (7k books)
  Training petaFLOPs: 18k
  Architecture: De
  Objective: LTR
  ```

### Vision Models

- **Eva02-E** [[BAAI]]() Mar. 2023 [[open]](https://github.com/huggingface/pytorch-image-models/tree/main)  
  EVA-02: A Visual Representation for Neon Genesis [[Preprint]](https://arxiv.org/abs/2303.11331v2)

  ```yaml
  Field: Vision-Language
  Params: 5B
  Training Data: 2B image-text pairs
  Architecture: Transformer
  Objective: MIM, Clip Constrastive
  ```

- **MAE->WSP-2B** [[Meta]]() Mar. 2023 [close]  
   The effectiveness of MAE pre-pretraining for billion-scale pretraining [[Preprint]](https://arxiv.org/abs/2303.13496)

  ```yaml
  Field: Vision
  Params: 6.5B
  Training Data: 3B images
  Architecture: Transformer
  Objective: MAE, Weakly-Supervised
  ```

- **OpenCLIP G/14** [[LAION]]() Mar. 2023 [[open]](https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s12B-b42K)

  ```yaml
  Field: Vision-Language
  Params: 2.5B
  Training Data: 2B images
  ```

- **ViT-22B** [[Google]]() Feb. 2023 [close]  
  [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442)

  ```yaml
  Field: Vision
  Params: 22B
  Training Data: 4B images
  Architecture: Transformer
  Objective: Supervised
  ```

- **ERNIE-ViLG** [[Baidu]](https://wenxin.baidu.com/wenxin/ernie-vilg) Dec. 2022 [close]  
   ERNIE-ViLG: Unified Generative Pre-training for Bidirectional Vision-Language Generation [[Preprint]](https://arxiv.org/abs/2112.15283)

  ```yaml
  Field: Image Generation (text to image)
  Params: 10B
  Training Data: 145M text-image pairs
  Architecture: Transformer, dVAE + De
  ```

- **InternImage-G** [[Shanghai AI Lab]](https://github.com/OpenGVLab/InternImage) Nov. 2022 [[open]](https://github.com/OpenGVLab/InternImage)
  InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions [[CVPR'23 Highlight]](https://arxiv.org/abs/2211.05778)

  ```yaml
  Field: Vision
  Params: 3B
  Architecture: CNN
  Core Operator: Deformable Convolution v3
  ```

- **Stable Diffusion** [[Stability AI]]() Aug. 2022 [[open]]()

  ```yaml
  Field: Image Generation (text to image)
  Params: 890M
  Training Data: 5B images
  Architecture: Transformer, Diffusion
  ```

- **Imagen** [[Google]](https://imagen.research.google/) May 2022  
   Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding [[Preprint]](https://arxiv.org/abs/2205.11487)

  ```yaml
  Field: Image Generation (text to image)
  Text Encoder: T5
  Image Decoder: Diffusion, Upsampler
  ```

- **Flamingo** [[DeepMind]]() Apr. 2022 [close]  
   Flamingo: a Visual Language Model for Few-Shot Learning [[Preprint]](https://arxiv.org/abs/2204.14198)

  ```yaml
  Field: Vision-Language
  Params: 80B
  ```

- **DALL·E 2** [[OpenAI]](https://openai.com/dall-e-2/) Apr. 2022  
   Hierarchical Text-Conditional Image Generation with CLIP Latents [[Preprint]](https://cdn.openai.com/papers/dall-e-2.pdf)

  ```yaml
  Field: Image Generation (text to image)
  Text Encoder: GPT2 (CLIP)
  Image Encoder: ViT (CLIP)
  Image Decoder: Diffusion, Upsampler
  ```

- **BaGuaLu** [[BAAI, Alibaba]]() Apr. 2022  
   BaGuaLu: targeting brain scale pretrained models with over 37 million cores [[PPoPP'22]](https://keg.cs.tsinghua.edu.cn/jietang/publications/PPOPP22-Ma%20et%20al.-BaGuaLu%20Targeting%20Brain%20Scale%20Pretrained%20Models%20w.pdf)

  ```yaml
  Field: Vision-Language
  Params: 174T
  Architecture: M6
  ```

- **SEER** [[Meta]]() Feb. 2022 [[open]](https://github.com/facebookresearch/vissl)  
   Vision Models Are More Robust And Fair When Pretrained On Uncurated Images Without Supervision [[Preprint]](https://arxiv.org/abs/2202.08360v2)

  ```yaml
  Field: Vision
  Params: 10B
  Training Data: 1B images
  Architecture: Convolution
  Objective: SwAV
  ```

- **ERNIE-ViLG** [[Baidu]](https://wenxin.baidu.com/wenxin/ernie-vilg) Dec. 2021  
   ERNIE-ViLG: Unified Generative Pre-training for Bidirectional Vision-Language Generation [[Preprint]](https://arxiv.org/abs/2112.15283)

  ```yaml
  Field: Image Generation (text to image)
  Params: 10B
  Training Data: 145M text-image pairs
  Architecture: Transformer, dVAE + De
  ```

- **NUWA** [[Microsoft]]() Nov. 2021 [[open]](https://github.com/microsoft/NUWA)  
   NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion [[Preprint]](https://arxiv.org/abs/2111.12417)

  ```yaml
  Field: Vision-Language
  Generatioon: Image, Video
  Params: 870M
  ```

- **SwinV2-G** [[Google]]() Nov. 2021 [[open]](https://github.com/microsoft/Swin-Transformer)  
   Swin Transformer V2: Scaling Up Capacity and Resolution [[CVPR'22]](https://arxiv.org/abs/2111.09883v2)

  ```yaml
  Field: Vision
  Params: 3B
  Training Data: 70M
  Architecture: Transformer
  Objective: Supervised
  ```

- **Zidongtaichu** [[CASIA]](http://www.ia.cas.cn/xwzx/kydt/202109/t20210927_6215538.html) Sept. 2021 [close]

  ```yaml
  Field: Image, Video, Language, Speech
  Params: 100B
  ```

- **ViT-G/14** [[Google]]() June 2021  
   Scaling Vision Transformers [[Preprint]](https://arxiv.org/abs/2106.04560)

  ```yaml
  Field: Vision
  Params: 1.8B
  Training Data: 300M images
  Training petaFLOPs: 3.4M
  Architecture: Transformer
  Objective: Supervised
  ```

- **CoAtNet** [[Google]](https://ai.googleblog.com/2021/09/toward-fast-and-accurate-neural.html) June 2021 [[open]](https://github.com/chinhsuanwu/coatnet-pytorch)  
   CoAtNet: Marrying Convolution and Attention for All Data Sizes [[NeurIPS'21]](https://arxiv.org/abs/2106.04803)

  ```yaml
  Field: Vision
  Params: 2.4B
  Training Data: 300M images
  Architecture: Transformer, Convolution
  Objective: Supervised
  ```

- **V-MoE** [[Google]](https://ai.googleblog.com/2022/01/scaling-vision-with-sparse-mixture-of.html) June 2021  
   Scaling Vision with Sparse Mixture of Experts [[NeurIPS'21]](https://proceedings.neurips.cc//paper/2021/file/48237d9f2dea8c74c2a72126cf63d933-Paper.pdf)

  ```yaml
  Field: Vision
  Params: 15B
  Training Data: 300M images
  Training Time: 16.8k TPUv3 days
  Training petaFLOPs: 33.9M
  Architecture: Transformer, MoE
  Objective: Supervised
  ```

- **CogView** [[BAAI, Alibaba]](https://wudao.aminer.cn/CogView/index.html) May 2021 [</>](https://github.com/THUDM/CogView)  
   CogView: Mastering Text-to-Image Generation via Transformers [[NeurIPS'21]](https://arxiv.org/abs/2105.13290)

  ```yaml
  Field: Vision-Language
  Params: 4B
  Training Data: 30M text-image pairs
  Training petaFLOPs: 27M
  Image Encoder: VAE
  Text Encoder & Image Decoder: GPT2
  ```

- **M6** [[Alibaba]](https://m6.aliyun.com/#/) Mar. 2021  
   M6: A Chinese Multimodal Pretrainer [[Preprint]](https://arxiv.org/abs/2103.00823)

  ```yaml
  Field: Vision-Language
  Params: 10T
  Training Data: 300G Texts + 2TB Images
  Training petaFLOPs: 5.5M
  Fusion: Single-stream
  Objective: MLM, IC
  ```

- **DALL·E** [[OpenAI]](https://openai.com/blog/dall-e/) Feb. 2021  
   Zero-Shot Text-to-Image Generation [[ICML'21]](https://arxiv.org/abs/2102.12092)

  ```yaml
  Field: Image Generation (text to image)
  Params: 12B
  Training Data: 250M text-images pairs
  Training petaFLOPs: 47M
  Image Encoder: dVAE
  Text Encoder & Image Decoder: GPT2
  ```

- **CLIP** [[OpenAI]](https://openai.com/blog/clip/) Jan. 2021  
   Learning Transferable Visual Models From Natural Language Supervision [[ICML'22]](https://arxiv.org/abs/2103.00020)

  ```yaml
  Field: Vision-Language
  Training Data: 400M text-image pairs
  Training petaFLOPs: 11M
  Image Encoder: ViT
  Text Encoder: GPT-2
  Fusion: Dual Encoder
  Objective: CMCL
  ```

- **ViT-H/14** [[Google]](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html) Oct. 2020 [[open]](https://github.com/google-research/vision_transformer)  
   An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale [[ICLR'20]](https://arxiv.org/abs/2010.11929)

  ```yaml
  Field: Vision
  Params: 632M
  Training Data: 300M images
  Training petaFLOPs: 13M
  Architecture: Transformer
  Objective: Supervised
  ```

- **iGPT-XL** [[OpenAI]](https://openai.com/blog/image-gpt/) June 2020 [[open]](https://github.com/openai/image-gpt)  
   Generative Pretraining From Pixels [[ICML'20]](https://proceedings.mlr.press/v119/chen20s.html)

  ```yaml
  Field: Image Generation
  Params: 6.8B
  Training Data: 1M images
  Training petaFLOPs: 33M
  Architecture: Transformer, De
  ```

- **BigGAN-deep** [[DeepMind]]() Sept. 2018 [[open]](https://github.com/ajbrock/BigGAN-PyTorch)  
   Large Scale GAN Training for High Fidelity Natural Image Synthesis [[ICLR'19]](https://arxiv.org/abs/1809.11096)

  ```yaml
  Field: Image Generation
  Params: 158M
  Training Data: 300M images
  Training petaFLOPs: 3M
  Architecture: Convolution, GAN
  Resolution: 512x512
  ```

### Reinforcement Learning

- **PaLM-E** [[Google]](https://palm-e.github.io/) March 2023 [close]  
   PaLM-E: An Embodied Multimodal Language Model [[Preprint]](https://palm-e.github.io/assets/palm-e.pdf)

  ```yaml
  Field: Reinforcement Learning
  Params: 562B (540B LLM + 22B Vi)
  ```

- **Gato** [[DeepMind]](https://www.deepmind.com/publications/a-generalist-agent) May 2022 [close]  
   A Generalist Agent [[Preprint]](https://arxiv.org/abs/2205.06175)

  ```yaml
  Field: Reinforcement Learning
  Params: 1.2B
  Training Data: (604 Tasks)
  Objective: Supervised
  ```

### Speech

- **USM** [[Google]](https://sites.research.google/usm/) Mar. 2023 [close]  
  Google USM: Scaling Automatic Speech Recognition Beyond 100 Languages [[Preprint]](https://arxiv.org/pdf/2303.01037v2.pdf)

  ```yaml
  Field: Speech
  Params: 2B
  Training Data: 12,000,000 hours
  ```

- **Whisper** [[OpenAI]](https://openai.com/research/whisper) Sept. 2022 [[close]](https://github.com/openai/whisper)  
   Robust Speech Recognition via Large-Scale Weak Supervision [[Preprint]](https://arxiv.org/pdf/2212.04356.pdf)

  ```yaml
  Field: Speech
  Params: 1.55B
  Training Data: 680,000 hours
  Objective: Weakly Supervised
  ```

- **HuBERT** [[Meta]](https://ai.facebook.com/blog/hubert-self-supervised-representation-learning-for-speech-recognition-generation-and-compression/) June 2021 [[open]](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert)  
   HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units [[Preprint]](https://arxiv.org/abs/2106.07447)

  ```yaml
  Field: Speech
  Params: 1B
  Training Data: 60,000 hours
  Objective: MLM
  ```

- **wav2vec 2.0** [[Meta]]() Oct. 2020 [[open]](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec)  
   wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations [[NeurIPS'20]](https://arxiv.org/abs/2006.11477)

  ```yaml
  Field: Speech
  Params: 317M
  Training Data: 50,000 hours
  Training petaFLOPs: 430M
  Objective: MLM
  ```

- **DeepSpeech 2** [[Meta]]() Dec. 2015 [[open]](https://github.com/PaddlePaddle/PaddleSpeech)  
   Deep Speech 2: End-to-End Speech Recognition in
  English and Mandarin [[ICML'15]](https://arxiv.org/pdf/1512.02595.pdf)

      ```yaml
      Field: Speech
      Params: 300M
      Training Data: 21,340 hours
      ```

## License

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
