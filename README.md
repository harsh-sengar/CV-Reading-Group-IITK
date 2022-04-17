# Computer Vision Reading Group (IITK)

## Table of Contents
- [Computer Vision Reading Group (IITK)](#computer-vision-reading-group-iitk)
  - [Table of Contents](#table-of-contents)
  - [Sessions](#sessions)
    - [Session 1: Harshvardhan Pratap Singh (14/12/2021)](#session-1-harshvardhan-pratap-singh-14122021)
      - [Image Inpainting via Conditional Texture and Structure Dual Generation (ICCV 2021)](#image-inpainting-via-conditional-texture-and-structure-dual-generation-iccv-2021)
    - [Session 2: Kranti Kumar Parida (21/12/2021)](#session-2-kranti-kumar-parida-21122021)
      - [Structure from Silence: Learning Scene Structure from Ambient Sound (CoRL 2021)](#structure-from-silence-learning-scene-structure-from-ambient-sound-corl-2021)
      - [Learning Audio-Visual Dereverberation (arxiv 2021)](#learning-audio-visual-dereverberation-arxiv-2021)
    - [Session 3: Ayush Pande (28/12/2021)](#session-3-ayush-pande-28122021)
      - [Image Style Transfer Using Convolutional Neural Networks (CVPR 2016)](#image-style-transfer-using-convolutional-neural-networks-cvpr-2016)
      - [Texture Synthesis Using Convolutional Neural Networks (NIPS 2015)](#texture-synthesis-using-convolutional-neural-networks-nips-2015)
      - [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (ICCV 2017)](#arbitrary-style-transfer-in-real-time-with-adaptive-instance-normalization-iccv-2017)
    - [Session 4: Prasen Kumar Sharma (04/01/2022)](#session-4-prasen-kumar-sharma-04012022)
      - [Mask Guided Matting via Progressive Refinement Network (CVPR2021)](#mask-guided-matting-via-progressive-refinement-network-cvpr2021)
      - [ZeroQ: A Novel Zero Shot Quantization Framework (CVPR 2020)](#zeroq-a-novel-zero-shot-quantization-framework-cvpr-2020)
    - [Session 5: Gaurav Sharma (11/01/2022)](#session-5-gaurav-sharma-11012022)
      - [Learning Video Stabilization Using Optical Flow (CVPR 2020)](#learning-video-stabilization-using-optical-flow-cvpr-2020)
      - [Hybrid Neural Fusion for Full-frame Video Stabilization (ICCV 2021)](#hybrid-neural-fusion-for-full-frame-video-stabilization-iccv-2021)
    - [Session 6: Neeraj Matiyali (25/01/2022)](#session-6-neeraj-matiyali-25012022)
      - [A Survey on Neural Speech Synthesis (arxiv 2021)](#a-survey-on-neural-speech-synthesis-arxiv-2021)
      - [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (ICLR 2021)](#fastspeech-2-fast-and-high-quality-end-to-end-text-to-speech-iclr-2021)
    - [Session 7: Harshvardhan Pratap Singh (01/02/2022)](#session-7-harshvardhan-pratap-singh-01022022)
      - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ICLR 2021)](#an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale-iclr-2021)
      - [Vision Transformers for Dense Prediction (ICCV 2021)](#vision-transformers-for-dense-prediction-iccv-2021)
    - [Session 8: Kranti Kumar Parida (08/02/2022)](#session-8-kranti-kumar-parida-08022022)
      - [Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation (SIGGRAPH 2018)](#looking-to-listen-at-the-cocktail-party-a-speaker-independent-audio-visual-model-for-speech-separation-siggraph-2018)
      - [VisualVoice: Audio-Visual Speech Separation With Cross-Modal Consistency (CVPR 2021)](#visualvoice-audio-visual-speech-separation-with-cross-modal-consistency-cvpr-2021)
    - [Session 9: Ayush Pande (17/02/2022)](#session-9-ayush-pande-17022022)
      - [AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer (ICCV 2021)](#adaattn-revisit-attention-mechanism-in-arbitrary-neural-style-transfer-iccv-2021)
      - [Video Autoencoder: self-supervised disentanglement of static 3D structure and motion (ICCV 2021)](#video-autoencoder-self-supervised-disentanglement-of-static-3d-structure-and-motion-iccv-2021)
    - [Session 10: Neeraj Matiyali (22/03/2022)](#session-10-neeraj-matiyali-22032022)
      - [Neural Voice Cloning with a Few Samples (NeurIPS 2018)](#neural-voice-cloning-with-a-few-samples-neurips-2018)
    - [Session 11: Harshvardhan Pratap Singh (29/03/2022)](#session-11-harshvardhan-pratap-singh-29032022)
      - [TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up (NeurIPS 2021)](#transgan-two-pure-transformers-can-make-one-strong-gan-and-that-can-scale-up-neurips-2021)
    - [Session 12: Kranti Kumar Parida (12/04/2022)](#session-12-kranti-kumar-parida-12042022)
      - [Audio-Visual Speech Codecs: Rethinking Audio-Visual Speech Enhancement by Re-Synthesis (arxiv 2022)](#audio-visual-speech-codecs-rethinking-audio-visual-speech-enhancement-by-re-synthesis-arxiv-2022)
      - [Visual Acoustic Matching (arxiv 2022)](#visual-acoustic-matching-arxiv-2022)


## Sessions

---

### Session 1: Harshvardhan Pratap Singh (14/12/2021)

- **Date/Time:** 14/12/2021, 11:00 IST
- **Presenter:** Harshvardhan Pratap Singh
- **Presentation:** [slides (pptx)](slides/session1_image_inpainting.pptx), [slides (pdf)](slides/session1_image_inpainting.pdf) 

#### Image Inpainting via Conditional Texture and Structure Dual Generation (ICCV 2021)

- **Abstract:** Deep generative approaches have recently made considerable progress in image inpainting by introducing structure priors. Due to the lack of proper interaction with image texture during structure reconstruction, however, current solutions are incompetent in handling the cases with large corruptions, and they generally suffer from distorted results. In this paper, we propose a novel two-stream network for image inpainting, which models the structure-constrained texture synthesis and texture-guided structure reconstruction in a coupled manner so that they better leverage each other for more plausible generation. Furthermore, to enhance the global consistency, a Bi-directional Gated Feature Fusion (Bi-GFF) module is designed to exchange and combine the structure and texture information and a Contextual Feature Aggregation (CFA) module is developed to refine the generated contents by region affinity learning and multi-scale feature aggregation. Qualitative and quantitative experiments on the CelebA, Paris StreetView and Places2 datasets demonstrate the superiority of the proposed method. Our code is available at https://github.com/Xiefan-Guo/CTSDG.
- **Paper Link:** [paper (cvf)](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_Image_Inpainting_via_Conditional_Texture_and_Structure_Dual_Generation_ICCV_2021_paper.html)



---

### Session 2: Kranti Kumar Parida (21/12/2021)

- **Date/Time:** 21/12/2021, 11:00 IST
- **Presenter:** Kranti Kumar Parida
- **Presentation:** [slides (pdf)](slides/session2_sound2depth_dereverberation.pdf)

#### Structure from Silence: Learning Scene Structure from Ambient Sound (CoRL 2021)
- **Abstract:** From whirling ceiling fans to ticking clocks, the sounds that we hear subtly vary as we move through a scene. We ask whether these ambient sounds convey information about 3D scene structure and, if so, whether they provide a useful learning signal for multimodal models. To study this, we collect a dataset of paired audio and RGB-D recordings from a variety of quiet indoor scenes. We then train models that estimate the distance to nearby walls, given only audio as input. We also use these recordings to learn multimodal representations through self-supervision, by training a network to associate images with their corresponding sounds. These results suggest that ambient sound conveys a surprising amount of information about scene structure, and that it is a useful signal for learning multimodal features.ZeroQ
- **Paper Link:** [paper (openreview)](https://openreview.net/forum?id=ht3aHpc1hUt)

#### Learning Audio-Visual Dereverberation (arxiv 2021)
- **Abstract:** Reverberation from audio reflecting off surfaces and objects in the environment not only degrades the quality of speech for human perception, but also severely impacts the accuracy of automatic speech recognition. Prior work attempts to remove reverberation based on the audio modality only. Our idea is to learn to dereverberate speech from audio-visual observations. The visual environment surrounding a human speaker reveals important cues about the room geometry, materials, and speaker location, all of which influence the precise reverberation effects in the audio stream. We introduce Visually-Informed Dereverberation of Audio (VIDA), an end-to-end approach that learns to remove reverberation based on both the observed sounds and visual scene. In support of this new task, we develop a large-scale dataset that uses realistic acoustic renderings of speech in real-world 3D scans of homes offering a variety of room acoustics. Demonstrating our approach on both simulated and real imagery for speech enhancement, speech recognition, and speaker identification, we show it achieves state-of-the-art performance and substantially improves over traditional audio-only methods. Project page: [this http URL](http://vision.cs.utexas.edu/projects/learning-audio-visual-dereverberation).
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/2106.07732)


---

### Session 3: Ayush Pande (28/12/2021)

- **Date/Time:** 28/12/2021, 11:00 IST
- **Presenter:** Ayush Pande
- **Presentation:** [slides (pptx)](slides/session3_style_transfer.pptx), [slides (pdf)](slides/session3_style_transfer.pdf)

#### Image Style Transfer Using Convolutional Neural Networks (CVPR 2016)
- **Abstract:** Rendering the semantic content of an image in different styles is a difficult image processing task. Arguably, a major limiting factor for previous approaches has been the lack of image representations that explicitly represent semantic information and, thus, allow to separate image content from style. Here we use image representations derived from Convolutional Neural Networks optimised for object recognition, which make high level image information explicit. We introduce A Neural Algorithm of Artistic Style that can separate and recombine the image content and style of natural images. The algorithm allows us to produce new images of high perceptual quality that combine the content of an arbitrary photograph with the appearance of numerous well-known artworks. Our results provide new insights into the deep image representations learned by Convolutional Neural Networks and demonstrate their potential for high level image synthesis and manipulation.
- **Paper Link:** [paper (cvf)](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)


#### Texture Synthesis Using Convolutional Neural Networks (NIPS 2015)
- **Abstract:** Here we introduce a new model of natural textures based on the feature spaces of convolutional neural networks optimised for object recognition. Samples from the model are of high perceptual quality demonstrating the generative power of neural networks trained in a purely discriminative fashion. Within the model, textures are represented by the correlations between feature maps in several layers of the network. We show that across layers the texture representations increasingly capture the statistical properties of natural images while making object information more and more explicit. The model provides a new tool to generate stimuli for neuroscience and might offer insights into the deep representations learned by convolutional neural networks.
- Paper Link: [paper (arxiv)](https://arxiv.org/abs/1505.07376)

#### Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization (ICCV 2017)
- **Abstract:** Gatys et al. recently introduced a neural algorithm that renders a content image in the style of another image, achieving so-called style transfer. However, their framework requires a slow iterative optimization process, which limits its practical application. Fast approximations with feed-forward neural networks have been proposed to speed up neural style transfer. Unfortunately, the speed improvement comes at a cost: the network is usually tied to a fixed set of styles and cannot adapt to arbitrary new styles. In this paper, we present a simple yet effective approach that for the first time enables arbitrary style transfer in real-time. At the heart of our method is a novel adaptive instance normalization (AdaIN) layer that aligns the mean and variance of the content features with those of the style features. Our method achieves speed comparable to the fastest existing approach, without the restriction to a pre-defined set of styles. In addition, our approach allows flexible user controls such as content-style trade-off, style interpolation, color & spatial controls, all using a single feed-forward neural network.
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/1703.06868)

---

### Session 4: Prasen Kumar Sharma (04/01/2022)

- **Date/Time:** 04/01/2022, 11:00 IST
- **Presenter:** Prasen Kumar Sharma
- **Presentation:** [slides (pdf)](slides/session4_image_matting_quantization.pdf)

#### Mask Guided Matting via Progressive Refinement Network (CVPR2021)
- Abstract: We propose Mask Guided (MG) Matting, a robust matting framework that takes a general coarse mask as guidance. MG Matting leverages a network (PRN) design which encourages the matting model to provide self-guidance to progressively refine the uncertain regions through the decoding process. A series of guidance mask perturbation operations are also introduced in the training to further enhance its robustness to external guidance. We show that PRN can generalize to unseen types of guidance masks such as trimap and low-quality alpha matte, making it suitable for various application pipelines. In addition, we revisit the foreground color prediction problem for matting and propose a surprisingly simple improvement to address the dataset issue. Evaluation on real and synthetic benchmarks shows that MG Matting achieves state-of-the-art performance using various types of guidance inputs. Code and models are available at https://github.com/yucornetto/MGMatting.
- Paper Link: [paper (cvf)](https://openaccess.thecvf.com/content/CVPR2021/html/Yu_Mask_Guided_Matting_via_Progressive_Refinement_Network_CVPR_2021_paper.html)

#### ZeroQ: A Novel Zero Shot Quantization Framework (CVPR 2020)
- **Abstract:** Abstract: Quantization is a promising approach for reducing the inference time and memory footprint of neural networks. However, most existing quantization methods require access to the original training dataset for retraining during quantization. This is often not possible for applications with sensitive or proprietary data, e.g., due to privacy and security concerns. Existing zero-shot quantization methods use different heuristics to address this, but they result in poor performance, especially when quantizing to ultralow precision. Here, we propose ZEROQ, a novel zeroshot quantization framework to address this. ZEROQ enables mixed-precision quantization without any access to the training or validation data. This is achieved by optimizing for a Distilled Dataset, which is engineered to match the statistics of batch normalization across different layers of the network. ZEROQ supports both uniform and mixed-precision quantization. For the latter, we introduce a novel Pareto frontier based method to automatically determine the mixed-precision bit setting for all layers, with no manual search involved. We extensively test our proposed method on a diverse set of models, including ResNet18/50/152, MobileNetV2, ShuffleNet, SqueezeNext, and InceptionV3 on ImageNet, as well as RetinaNet-ResNet50 on the Microsoft COCO dataset. In particular, we show that ZEROQ can achieve 1.71% higher accuracy on MobileNetV2, as compared to the recently proposed DFQ [32] method. Importantly, ZEROQ has a very low computational overhead, and it can finish the entire quantization process in less than 30s (0.5% of one epoch training time of ResNet50 on ImageNet). [We have open-sourced the ZEROQ framework.](https://github.com/amirgholami/ZeroQ)
- **Paper Link:** [paper (cvf)](https://openaccess.thecvf.com/content_CVPR_2020/html/Cai_ZeroQ_A_Novel_Zero_Shot_Quantization_Framework_CVPR_2020_paper.html)

---

### Session 5: Gaurav Sharma (11/01/2022)

- **Date/Time:** 11/01/2022, 11:00 IST
- **Presenter:** Gaurav Sharma
- **Presentation:** 
  
#### Learning Video Stabilization Using Optical Flow (CVPR 2020)
- **Abstract:** We propose a novel neural network that infers the per-pixel warp fields for video stabilization from the optical flow fields of the input video. While previous learning based video stabilization methods attempt to implicitly learn frame motions from color videos, our method resorts to optical flow for motion analysis and directly learns the stabilization using the optical flow. We also propose a pipeline that uses optical flow principal components for motion inpainting and warp field smoothing, making our method robust to moving objects, occlusion and optical flow inaccuracy, which is challenging for other video stabilization methods. Our method achieves quantitatively and visually better results than the state-of-the-art optimization based and deep learning based video stabilization methods. Our method also gives a 3x speed improvement compared to the optimization based methods.
- **Paper Link:** [paper (cvf)](https://openaccess.thecvf.com/content_CVPR_2020/html/Yu_Learning_Video_Stabilization_Using_Optical_Flow_CVPR_2020_paper.html)

#### Hybrid Neural Fusion for Full-frame Video Stabilization (ICCV 2021)

- **Abstract:** Existing video stabilization methods often generate vis- ible distortion or require aggressive cropping of frame boundaries, resulting in smaller field of views. In this work, we present a frame synthesis algorithm to achieve full-frame video stabilization. We first estimate dense warp fields from neighboring frames and then synthesize the stabilized frame by fusing the warped contents. Our core technical novelty lies in the learning-based hybrid-space fusion that allevi- ates artifacts caused by optical flow inaccuracy and fast- moving objects. We validate the effectiveness of our method on the NUS, selfie, and DeepStab video datasets. Extensive experiment results demonstrate the merits of our approach over prior video stabilization methods.
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/2102.06205)

---

### Session 6: Neeraj Matiyali (25/01/2022)

- **Date/Time:** 25/01/2022, 11:00 IST
- **Presenter:** Neeraj Matiyali
- **Presentation:** [slides (pdf)](slides/session6_text_to_speech.pdf)

#### A Survey on Neural Speech Synthesis (arxiv 2021)
- **Abstract:** Text to speech (TTS), or speech synthesis, which aims to synthesize intelligible and natural speech given text, is a hot research topic in speech, language, and machine learning communities and has broad applications in the industry. As the development of deep learning and artificial intelligence, neural network-based TTS has significantly improved the quality of synthesized speech in recent years. In this paper, we conduct a comprehensive survey on neural TTS, aiming to provide a good understanding of current research and future trends. We focus on the key components in neural TTS, including text analysis, acoustic models and vocoders, and several advanced topics, including fast TTS, low-resource TTS, robust TTS, expressive TTS, and adaptive TTS, etc. We further summarize resources related to TTS (e.g., datasets, opensource implementations) and discuss future research directions. This survey can serve both academic researchers and industry practitioners working on TTS.
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/2106.15561)

#### FastSpeech 2: Fast and High-Quality End-to-End Text to Speech (ICLR 2021)
- **Abstract:** Non-autoregressive text to speech (TTS) models such as FastSpeech can synthesize speech significantly faster than previous autoregressive models with comparable quality. The training of FastSpeech model relies on an autoregressive teacher model for duration prediction (to provide more information as input) and knowledge distillation (to simplify the data distribution in output), which can ease the one-to-many mapping problem (i.e., multiple speech variations correspond to the same text) in TTS. However, FastSpeech has several disadvantages: 1) the teacher-student distillation pipeline is complicated and time-consuming, 2) the duration extracted from the teacher model is not accurate enough, and the target mel-spectrograms distilled from teacher model suffer from information loss due to data simplification, both of which limit the voice quality. In this paper, we propose FastSpeech 2, which addresses the issues in FastSpeech and better solves the one-to-many mapping problem in TTS by 1) directly training the model with ground-truth target instead of the simplified output from teacher, and 2) introducing more variation information of speech (e.g., pitch, energy and more accurate duration) as conditional inputs. Specifically, we extract duration, pitch and energy from speech waveform and directly take them as conditional inputs in training and use predicted values in inference. We further design FastSpeech 2s, which is the first attempt to directly generate speech waveform from text in parallel, enjoying the benefit of fully end-to-end inference. Experimental results show that 1) FastSpeech 2 achieves a 3x training speed-up over FastSpeech, and FastSpeech 2s enjoys even faster inference speed; 2) FastSpeech 2 and 2s outperform FastSpeech in voice quality, and FastSpeech 2 can even surpass autoregressive models. Audio samples are available at https://speechresearch.github.io/fastspeech2/.
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/2006.04558)

---

### Session 7: Harshvardhan Pratap Singh (01/02/2022)

- **Date/Time:** 01/02/2022, 11:00 IST
- **Presenter:** Harshvardhan Pratap Singh
- **Presentation:** [slides (pdf)](slides/session7_vision_transformer.pdf)

#### An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ICLR 2021)

- **Abstract:** While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/2010.11929)

#### Vision Transformers for Dense Prediction (ICCV 2021)
- **Abstract:** We introduce dense vision transformers, an architecture that leverages vision transformers in place of convolutional networks as a backbone for dense prediction tasks. We assemble tokens from various stages of the vision transformer into image-like representations at various resolutions and progressively combine them into full-resolution predictions using a convolutional decoder. The transformer backbone processes representations at a constant and relatively high resolution and has a global receptive field at every stage. These properties allow the dense vision transformer to provide finer-grained and more globally coherent predictions when compared to fully-convolutional networks. Our experiments show that this architecture yields substantial improvements on dense prediction tasks, especially when a large amount of training data is available. For monocular depth estimation, we observe an improvement of up to 28% in relative performance when compared to a state-of-the-art fully-convolutional network. When applied to semantic segmentation, dense vision transformers set a new state of the art on ADE20K with 49.02% mIoU. We further show that the architecture can be fine-tuned on smaller datasets such as NYUv2, KITTI, and Pascal Context where it also sets the new state of the art. Our models are available at [this https URL](https://github.com/intel-isl/DPT).
- **Paper Link:** [paper (cvf)](https://arxiv.org/abs/2103.13413)

--- 

### Session 8: Kranti Kumar Parida (08/02/2022)

- **Date/Time:** 08/02/2022, 11:00 IST
- **Presenter:** Kranti Kumar Parida
- **Presentation:** 

#### Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech Separation (SIGGRAPH 2018)
- **Abstract:** We present a joint audio-visual model for isolating a single speech signal from a mixture of sounds such as other speakers and background noise. Solving this task using only audio as input is extremely challenging and does not provide an association of the separated speech signals with speakers in the video. In this paper, we present a deep network-based model that incorporates both visual and auditory signals to solve this task. The visual features are used to "focus" the audio on desired speakers in a scene and to improve the speech separation quality. To train our joint audio-visual model, we introduce AVSpeech, a new dataset comprised of thousands of hours of video segments from the Web. We demonstrate the applicability of our method to classic speech separation tasks, as well as real-world scenarios involving heated interviews, noisy bars, and screaming children, only requiring the user to specify the face of the person in the video whose speech they want to isolate. Our method shows clear advantage over state-of-the-art audio-only speech separation in cases of mixed speech. In addition, our model, which is speaker-independent (trained once, applicable to any speaker), produces better results than recent audio-visual speech separation methods that are speaker-dependent (require training a separate model for each speaker of interest).
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/1804.03619)

#### VisualVoice: Audio-Visual Speech Separation With Cross-Modal Consistency (CVPR 2021)
- **Abstract:** We introduce a new approach for audio-visual speech separation. Given a video, the goal is to extract the speech associated with a face in spite of simultaneous background sounds and/or other human speakers. Whereas existing methods focus on learning the alignment between the speaker's lip movements and the sounds they generate, we propose to leverage the speaker's face appearance as an additional prior to isolate the corresponding vocal qualities they are likely to produce. Our approach jointly learns audio-visual speech separation and cross-modal speaker embeddings from unlabeled video. It yields state-of-the-art results on five benchmark datasets for audio-visual speech separation and enhancement, and generalizes well to challenging real-world videos of diverse scenarios. Our video results and code: http://vision.cs.utexas.edu/projects/VisualVoice/.
- **Paper Link:** [paper (cvf)](https://openaccess.thecvf.com/content/CVPR2021/html/Gao_VisualVoice_Audio-Visual_Speech_Separation_With_Cross-Modal_Consistency_CVPR_2021_paper.html)

---

### Session 9: Ayush Pande (17/02/2022)

- **Date/Time:** 17/02/2022, 11:00 IST
- **Presenter:** Ayush Pande
- **Presentation:** [slides (pdf)](slides/session9_adaattn_video_ae.pdf)

#### AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer (ICCV 2021)
- **Abstract:** Fast arbitrary neural style transfer has attracted widespread attention from academic, industrial and art communities due to its flexibility in enabling various applications. Existing solutions either attentively fuse deep style feature into deep content feature without considering feature distributions, or adaptively normalize deep content feature according to the style such that their global statistics are matched. Although effective, leaving shallow feature unexplored and without locally considering feature statistics, they are prone to unnatural output with unpleasing local distortions. To alleviate this problem, in this paper, we propose a novel attention and normalization module, named Adaptive Attention Normalization (AdaAttN), to adaptively perform attentive normalization on per-point basis. Specifically, spatial attention score is learnt from both shallow and deep features of content and style images. Then per-point weighted statistics are calculated by regarding a style feature point as a distribution of attention-weighted output of all style feature points. Finally, the content feature is normalized so that they demonstrate the same local feature statistics as the calculated per-point weighted style feature statistics. Besides, a novel local feature loss is derived based on AdaAttN to enhance local 
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/2108.03647)

#### Video Autoencoder: self-supervised disentanglement of static 3D structure and motion (ICCV 2021)
- **Abstract:** A video autoencoder is proposed for learning disentangled representations of 3D structure and camera pose from videos in a self-supervised manner. Relying on temporal continuity in videos, our work assumes that the 3D scene structure in nearby video frames remains static. Given a sequence of video frames as input, the video autoencoder extracts a disentangled representation of the scene includ- ing: (i) a temporally-consistent deep voxel feature to represent the 3D structure and (ii) a 3D trajectory of camera pose for each frame. These two representations will then be re-entangled for rendering the input video frames. This video autoencoder can be trained directly using a pixel reconstruction loss, without any ground truth 3D or camera pose annotations. The disentangled representation can be applied to a range of tasks, including novel view synthesis, camera pose estimation, and video generation by motion following. We evaluate our method on several large- scale natural video datasets, and show generalization results on out-of-domain images.
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/2110.02951)

----


### Session 10: Neeraj Matiyali (22/03/2022)

- **Date/Time:** 23/03/2022, 11:00 IST
- **Presenter:** Neeraj Matiyali
- **Presentation:** [slides (pdf)](slides/session10_voice_cloning.pdf)

#### Neural Voice Cloning with a Few Samples (NeurIPS 2018)

- **Abstract:** Voice cloning is a highly desired feature for personalized speech interfaces. Neural network based speech synthesis has been shown to generate high quality speech for a large number of speakers. In this paper, we introduce a neural voice cloning system that takes a few audio samples as input. We study two approaches: speaker adaptation and speaker encoding. Speaker adaptation is based on fine-tuning a multi-speaker generative model with a few cloning samples. Speaker encoding is based on training a separate model to directly infer a new speaker embedding from cloning audios and to be used with a multi-speaker generative model. In terms of naturalness of the speech and its similarity to original speaker, both approaches can achieve good performance, even with very few cloning audios. While speaker adaptation can achieve better naturalness and similarity, the cloning time or required memory for the speaker encoding approach is significantly less, making it favorable for low-resource deployment.
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/1802.06006v3)


----

### Session 11: Harshvardhan Pratap Singh (29/03/2022)

- **Date/Time:** 29/03/2022, 11:00 IST
- **Presenter:** Harshvardhan Pratap Singh
- **Presentation:** [slides (pdf)](slides/session11_transgan.pdf)

#### TransGAN: Two Pure Transformers Can Make One Strong GAN, and That Can Scale Up (NeurIPS 2021)

- Abstract: The recent explosive interest on transformers has suggested their potential to become powerful "universal" models for computer vision tasks, such as classification, detection, and segmentation. While those attempts mainly study the discriminative models, we explore transformers on some more notoriously difficult vision tasks, e.g., generative adversarial networks (GANs). Our goal is to conduct the first pilot study in building a GAN completely free of convolutions, using only pure transformer-based architectures. Our vanilla GAN architecture, dubbed TransGAN, consists of a memory-friendly transformer-based generator that progressively increases feature resolution, and correspondingly a multi-scale discriminator to capture simultaneously semantic contexts and low-level textures. On top of them, we introduce the new module of grid self-attention for alleviating the memory bottleneck further, in order to scale up TransGAN to high-resolution generation. We also develop a unique training recipe including a series of techniques that can mitigate the training instability issues of TransGAN, such as data augmentation, modified normalization, and relative position encoding. Our best architecture achieves highly competitive performance compared to current state-of-the-art GANs using convolutional backbones. Specifically, TransGAN sets new state-of-the-art inception score of 10.43 and FID of 18.28 on STL-10, outperforming StyleGAN-V2. When it comes to higher-resolution (e.g. 256 x 256) generation tasks, such as on CelebA-HQ and LSUN-Church, TransGAN continues to produce diverse visual examples with high fidelity and impressive texture details. In addition, we dive deep into the transformer-based generation models to understand how their behaviors differ from convolutional ones, by visualizing training dynamics. The code is available at [this https URL](https://github.com/VITA-Group/TransGAN).
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/2102.07074)

----

### Session 12: Kranti Kumar Parida (12/04/2022)

- **Date/Time:** 12/04/2022, 11:00 IST
- **Presenter:** Kranti Kumar Parida
- **Presentation:** 
  
#### Audio-Visual Speech Codecs: Rethinking Audio-Visual Speech Enhancement by Re-Synthesis (arxiv 2022)
- **Abstract:** Since facial actions such as lip movements contain significant information about speech content, it is not surprising that audio-visual speech enhancement methods are more accurate than their audio-only counterparts. Yet, state-of-the-art approaches still struggle to generate clean, realistic speech without noise artifacts and unnatural distortions in challenging acoustic environments. In this paper, we propose a novel audio-visual speech enhancement framework for high-fidelity telecommunications in AR/VR. Our approach leverages audio-visual speech cues to generate the codes of a neural speech codec, enabling efficient synthesis of clean, realistic speech from noisy signals. Given the importance of speaker-specific cues in speech, we focus on developing personalized models that work well for individual speakers. We demonstrate the efficacy of our approach on a new audio-visual speech dataset collected in an unconstrained, large vocabulary setting, as well as existing audio-visual datasets, outperforming speech enhancement baselines on both quantitative metrics and human evaluation studies.
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/2203.17263)

#### Visual Acoustic Matching (arxiv 2022)
- **Abstract:** We introduce the visual acoustic matching task, in which an audio clip is transformed to sound like it was recorded in a target environment. Given an image of the target environment and a waveform for the source audio, the goal is to resynthesize the audio to match the target room acoustics as suggested by its visible geometry and materials. To address this novel task, we propose a cross-modal transformer model that uses audio-visual attention to inject visual properties into the audio and generate realistic audio output. In addition, we devise a self-supervised training objective that can learn acoustic matching from in-the-wild Web videos, despite their lack of acoustically mismatched audio. We demonstrate that our approach successfully translates human speech to a variety of real-world environments depicted in images, outperforming both traditional acoustic matching and more heavily supervised baselines.
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/2202.06875)

---