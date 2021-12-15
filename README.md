 # Computer Vision Reading Group (IITK)



 ## Schedule

 | Session No. | Date       | Time      | Presenter                 | Paper Title                                                            | Conf/Year | Links |
|-----------|------------|-----------|---------------------------|------------------------------------------------------------------------|-----------|-------|
| 1 | 14/12/2021 | 11:00 IST | Harshvardhan Pratap Singh | [Image Inpainting via Conditional Texture and Structure Dual Generation](#session-1) | ICCV 2021 |  [paper (cvf)](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_Image_Inpainting_via_Conditional_Texture_and_Structure_Dual_Generation_ICCV_2021_paper.html), [pptx](slides/session1_image_inpainting.pptx), [pdf](slides/session1_image_inpainting.pdf)      |
|2 | 21/12/2021 | 11:00 IST | Kranti Kumar Parida       | TBD                                                                    |           |       |

----
## Sessions

### Session 1

- **Date/Time:** 14/12/2021, 11:00 IST
- **Presenter:** Harshvardhan Pratap Singh

#### Image Inpainting via Conditional Texture and Structure Dual Generation

- **Abstract:** Deep generative approaches have recently made considerable progress in image inpainting by introducing structure priors. Due to the lack of proper interaction with image texture during structure reconstruction, however, current solutions are incompetent in handling the cases with large corruptions, and they generally suffer from distorted results. In this paper, we propose a novel two-stream network for image inpainting, which models the structure-constrained texture synthesis and texture-guided structure reconstruction in a coupled manner so that they better leverage each other for more plausible generation. Furthermore, to enhance the global consistency, a Bi-directional Gated Feature Fusion (Bi-GFF) module is designed to exchange and combine the structure and texture information and a Contextual Feature Aggregation (CFA) module is developed to refine the generated contents by region affinity learning and multi-scale feature aggregation. Qualitative and quantitative experiments on the CelebA, Paris StreetView and Places2 datasets demonstrate the superiority of the proposed method. Our code is available at https://github.com/Xiefan-Guo/CTSDG.
- **Paper Link:** [paper (cvf)](https://openaccess.thecvf.com/content/ICCV2021/html/Guo_Image_Inpainting_via_Conditional_Texture_and_Structure_Dual_Generation_ICCV_2021_paper.html)
- **Presentation Links:** [pptx](slides/session1_image_inpainting.pptx), [pdf](slides/session1_image_inpainting.pdf) 
