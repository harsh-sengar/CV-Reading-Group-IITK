 # Computer Vision Reading Group (IITK)



 ## Schedule

 | Session No. | Date       | Time      | Presenter                 | Paper Title                                                            | Conf/Year | Links |
|-----------|------------|-----------|---------------------------|------------------------------------------------------------------------|-----------|-------|
| 1 | 14/12/2021 | 11:00 IST | Harshvardhan Pratap Singh | [Image Inpainting via Conditional Texture and Structure Dual Generation](#session-1) | ICCV 2021 |  [paper (cvf)](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_Image_Inpainting_via_Conditional_Texture_and_Structure_Dual_Generation_ICCV_2021_paper.html), [slides (pptx)](slides/session1_image_inpainting.pptx), [slides (pdf)](slides/session1_image_inpainting.pdf)      |
|2 | 21/12/2021 | 11:00 IST | Kranti Kumar Parida       | [(1) Structure from Silence: Learning Scene Structure from Ambient Sound; (2) Learning Audio-Visual Dereverberation](#session-2)                                                                    |  (1) CoRL 2021; (2) arxiv 2021         |   [(1) paper (openreview)](https://openreview.net/forum?id=ht3aHpc1hUt), [(2) paper (arxiv)](https://arxiv.org/abs/2106.07732), [slides (pdf)](slides/session2_sound2depth_dereverberation.pdf)   |
|3 | 28/12/2021 | 11:00 IST | Ayush Pande       | [(1) Image Style Transfer Using Convolutional Neural Networks; (2) Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](#session-3)                                                                    |  (1) CVPR 2016; (2) ICCV 2017         |   [(1) paper (cvf)](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html), [(2) paper (arxiv)](https://arxiv.org/abs/1703.06868)  |
----
## Sessions
---

### Session 1

- **Date/Time:** 14/12/2021, 11:00 IST
- **Presenter:** Harshvardhan Pratap Singh
- **Presentation:** [slides (pptx)](slides/session1_image_inpainting.pptx), [slides (pdf)](slides/session1_image_inpainting.pdf) 

#### Image Inpainting via Conditional Texture and Structure Dual Generation

- **Abstract:** Deep generative approaches have recently made considerable progress in image inpainting by introducing structure priors. Due to the lack of proper interaction with image texture during structure reconstruction, however, current solutions are incompetent in handling the cases with large corruptions, and they generally suffer from distorted results. In this paper, we propose a novel two-stream network for image inpainting, which models the structure-constrained texture synthesis and texture-guided structure reconstruction in a coupled manner so that they better leverage each other for more plausible generation. Furthermore, to enhance the global consistency, a Bi-directional Gated Feature Fusion (Bi-GFF) module is designed to exchange and combine the structure and texture information and a Contextual Feature Aggregation (CFA) module is developed to refine the generated contents by region affinity learning and multi-scale feature aggregation. Qualitative and quantitative experiments on the CelebA, Paris StreetView and Places2 datasets demonstrate the superiority of the proposed method. Our code is available at https://github.com/Xiefan-Guo/CTSDG.
- **Paper Link:** [paper (cvf)](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_Image_Inpainting_via_Conditional_Texture_and_Structure_Dual_Generation_ICCV_2021_paper.html)



---

### Session 2

- **Date/Time:** 21/12/2021, 11:00 IST
- **Presenter:** Kranti Kumar Parida
- **Presentation:** [slides (pdf)](slides/session2_sound2depth_dereverberation.pdf)

#### Structure from Silence: Learning Scene Structure from Ambient Sound
- **Abstract:** From whirling ceiling fans to ticking clocks, the sounds that we hear subtly vary as we move through a scene. We ask whether these ambient sounds convey information about 3D scene structure and, if so, whether they provide a useful learning signal for multimodal models. To study this, we collect a dataset of paired audio and RGB-D recordings from a variety of quiet indoor scenes. We then train models that estimate the distance to nearby walls, given only audio as input. We also use these recordings to learn multimodal representations through self-supervision, by training a network to associate images with their corresponding sounds. These results suggest that ambient sound conveys a surprising amount of information about scene structure, and that it is a useful signal for learning multimodal features.
- **Paper Link:** [paper (openreview)](https://openreview.net/forum?id=ht3aHpc1hUt)

#### Learning Audio-Visual Dereverberation
- **Abstract:** Reverberation from audio reflecting off surfaces and objects in the environment not only degrades the quality of speech for human perception, but also severely impacts the accuracy of automatic speech recognition. Prior work attempts to remove reverberation based on the audio modality only. Our idea is to learn to dereverberate speech from audio-visual observations. The visual environment surrounding a human speaker reveals important cues about the room geometry, materials, and speaker location, all of which influence the precise reverberation effects in the audio stream. We introduce Visually-Informed Dereverberation of Audio (VIDA), an end-to-end approach that learns to remove reverberation based on both the observed sounds and visual scene. In support of this new task, we develop a large-scale dataset that uses realistic acoustic renderings of speech in real-world 3D scans of homes offering a variety of room acoustics. Demonstrating our approach on both simulated and real imagery for speech enhancement, speech recognition, and speaker identification, we show it achieves state-of-the-art performance and substantially improves over traditional audio-only methods. Project page: [this http URL](http://vision.cs.utexas.edu/projects/learning-audio-visual-dereverberation).
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/2106.07732)


---

### Session 3

- **Date/Time:** 28/12/2021, 11:00 IST
- **Presenter:** Ayush Pande
- **Presentation:** ---

#### Image Style Transfer Using Convolutional Neural Networks
- **Abstract:** Rendering the semantic content of an image in different styles is a difficult image processing task. Arguably, a major limiting factor for previous approaches has been the lack of image representations that explicitly represent semantic information and, thus, allow to separate image content from style. Here we use image representations derived from Convolutional Neural Networks optimised for object recognition, which make high level image information explicit. We introduce A Neural Algorithm of Artistic Style that can separate and recombine the image content and style of natural images. The algorithm allows us to produce new images of high perceptual quality that combine the content of an arbitrary photograph with the appearance of numerous well-known artworks. Our results provide new insights into the deep image representations learned by Convolutional Neural Networks and demonstrate their potential for high level image synthesis and manipulation.
- **Paper Link:** [paper (cvf)](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html)

#### Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization
- **Abstract:** Gatys et al. recently introduced a neural algorithm that renders a content image in the style of another image, achieving so-called style transfer. However, their framework requires a slow iterative optimization process, which limits its practical application. Fast approximations with feed-forward neural networks have been proposed to speed up neural style transfer. Unfortunately, the speed improvement comes at a cost: the network is usually tied to a fixed set of styles and cannot adapt to arbitrary new styles. In this paper, we present a simple yet effective approach that for the first time enables arbitrary style transfer in real-time. At the heart of our method is a novel adaptive instance normalization (AdaIN) layer that aligns the mean and variance of the content features with those of the style features. Our method achieves speed comparable to the fastest existing approach, without the restriction to a pre-defined set of styles. In addition, our approach allows flexible user controls such as content-style trade-off, style interpolation, color & spatial controls, all using a single feed-forward neural network.
- **Paper Link:** [paper (arxiv)](https://arxiv.org/abs/1703.06868)