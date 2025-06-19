# Response of Submission ID 2745

## Dear Reviewer W5Yy:
*Rating (8); Technical Quality (8); Presentation Quality (8)*

Thank you very much for your encouraging suggestions. **Your comments about the scalability, overhead, analysis for detection and different attack methods are very constructive.** The detailed responses are listed as follows:

**[Q1]** The OSN platforms might have a lot of variability across different strengths of the transformation. 

**[A1]** Thank you for your insightful suggestions. We agree with your point of view that image data could involve different transformations when transmitted on different OSN platforms. As we introduced in Section 3.2 and Figure 3 of the manuscript, Facebook, QQ, and WeChat have different Quality Factors (QF) for JPEG image compression. However, on the one hand, the calculation of the QF does not involve reverse-engineering operations, and we can calculate the QF based on the standard quantization tables for luminance (table 0) and chrominance (table 1) defined by the JPEG standard (ISO/IEC 10918-1) (the specific matrix values ​​can be found in the online anonymous github repository). On the other hand, we acknowledge that different OSN platforms may have more complex operation processes, and we do not oppose this scenario (we will add the discussion of considering more platforms in the final version), so in the manuscript we separate OSN to facilitate horizontal expansion of the simulation of more Online Social Networks. Nevertheless, we additionally set the OSN network to a more extreme QF value (and set QF to a random dynamic strategy between [0, 100]) to explore the effect of Stealthy-AE when the simulated OSN and the real test network have different QF values. The results (as the Table below) show that despite the setting of extreme QF values, Stealthy-AE still generates more than 90% of OSN-AE (significantly outperforms other baselines in Figure 5), which demonstrates the practicality and stability of our proposal.

| OSN      | Setting | OSN-AE (%) | Vanilla-AE (%) | Both-AE (%) | Non-AE (%) |
| -------- | ---------- | ---------- | -------------- | ----------- | ---------- |
| Facebook | QF=50      | 93.01      | 0.00           | 3.12        | 3.97       |
| Facebook | QF=30      | 91.42      | 0.00           | 3.85        | 4.73       |
| Facebook | QF=10      | 90.16      | 0.00           | 4.51        | 5.33       |
| Facebook | QF=Dynamic | 93.62      | 0.00           | 2.79        | 3.59       |
| QQ       | QF=50      | 96.54      | 0.00           | 1.02        | 2.44       |
| QQ       | QF=30      | 94.76      | 0.00           | 2.66        | 3.58       |
| QQ       | QF=10      | 91.35      | 0.00           | 3.23        | 5.42       |
| QQ       | QF=Dynamic | 96.93      | 0.00           | 1.35        | 1.72       |
| WeChat   | QF=50      | 97.06      | 0.00           | 0.54        | 2.40       |
| WeChat   | QF=30      | 94.23      | 0.00           | 1.16        | 4.61       |
| WeChat   | QF=10      | 92.91      | 0.00           | 2.20        | 4.89       |
| WeChat   | QF=Dynamic | 97.12      | 0.00           | 1.36        | 1.52       |

**[Q2]** About the overhead. 

**[A2]** Thanks for your constructive comments on the computational cost of the diffusion model and its applicability in real-time or time-constrained adversarial scenarios. To address this concern, we measured the average inference time and memory overhead per image (batch size = 1) across all methods on the same GPU setup (NVIDIA RTX 3090). The results are summarized below:

| **Method**           | **Inference Time (s)** | **Memory Footprint (MB)** | **Additional Overhead**     | **Remarks**                                  |
| -------------------- | ---------------------- | ------------------------- | --------------------------- | -------------------------------------------- |
| Base (FGSM)          | 0.03                   | 300                       | None                        | Standard FGSM/PGD implementation             |
| JPEGR \[38]          | 0.10                   | 320                       | JPEG compression simulation | Simple JPEG encoder model                    |
| UAP \[29]            | 0.07                   | 330                       | Pretrained universal noise  | Offline training needed                      |
| FIA \[52]            | 0.25                   | 380                       | Fourier transform           | Adds FFT memory overhead                     |
| ComModel \[33]       | 0.21                   | 450                       | Compressed feature encoder  | Dual-path feature pipeline                   |
| ComGAN \[44]         | 0.51                   | 580                       | GAN inference engine        | Multi-module architecture                    |
| SIO \[23]            | 0.45                   | 500                       | OSN simulation module       | Multi-stage pipeline                         |
| **Ours (Diffusion)** | **0.47**               | **480**                   | Fast diffusion (e.g., DDIM) | Efficient, end-to-end, OSN-robust simulation |

The results indicate that the diffusion model indeed our method indeed introduces additional computational cost, a slightly longer and acceptable inference time (0.47s). Even though, it maintains a **moderate memory footprint (480MB)**, which is comparable to ComModel [33] and **lower than ComGAN [44] and SIO [23]**. This can be attributed to the fact that they rely on complex multi-stage architectures or adversarial simulation modules. This demonstrates that our approach is **not inherently more resource-intensive in terms of space**. In contrast, the diffusion network can proactively model the transformation behavior of OSNs, enabling our model to simulate and adapt to OSN-induced perturbations in advance and produce more robust adversarial examples. Furthermore, while diffusion models are often viewed as computationally heavy, their footprint is largely determined by the number of sampling steps, which can be significantly reduced using techniques like **DDIM, FastDiff, or distillation-based approximations**. We plan to incorporate such strategies in future optimizations and discuss them in the paper. Overall, our method offers a **strong trade-off between time, space, and robustness**, making it highly practical for OSN-facing adversarial applications.

**[Q3]** About the discussion around how to detect. 

**[A3]** We appreciate the reviewer’s suggestion regarding the discussion on defenses. In fact, we have conducted an extensive evaluation of 7 representative adversarial example (AE) detection methods, as summarized in Table 3 of the manuscript. These include detection strategies based on image transformations (TT [42]), feature squeezing (FS [48]), data manifold analysis (MagNet [28]), rejection mechanisms (RR [32]), and out-of-distribution detection (EBD [24]). Additionally, we evaluated two recent detection frameworks specifically tailored for OSN-transmitted images: PRest-Net [5] and IFD [46]. Our results demonstrate that Stealthy-AE consistently achieves the lowest detection rates across all detectors, with the maximum detection rate being only 40.1% (by IFD) and as low as 31.9% (by EBD). This strongly supports the claim that our OSN-aware AEs remain stealthy even under modern detection schemes, including those specifically designed to handle compression-related transformations. We have also discussed these observations in Section 4.4. To further improve defense capabilities, future work could explore joint adversarial training with simulated OSN pipelines to anticipate transmission-induced transformations. Additionally, temporal consistency checks across multi-session uploads or cross-platform image fingerprinting may provide promising directions for detecting stealthy AEs. 

**[Q4]** About the discussion around how to detect. 

**[A4]** We agree with the reviewer that in black-box or query-efficient settings, such as NES and SPSA, the attack success rates are relatively lower compared to white-box attacks, as also reflected in Table 4. This is consistent with known limitations of these methods. However, we emphasize that Stealthy-AE is a general framework designed to be attack-agnostic, meaning it can be readily combined with any future adversarial attack methods, including more effective query-based or transfer-based techniques as they evolve. Our contribution is orthogonal to the attack generation mechanism, Stealthy-AE focuses on concealing AEs during OSN transmission, and thus can serve as a powerful enhancement layer for any attack strategy. We have incorporated these considerations into the Discussion section of the final version. 


## Dear Reviewer 7Hkb:
*Rating (5); Technical Quality (6); Presentation Quality (5)*

Thank you for your useful comments and suggestions on our manuscript. **We have studied the comments carefully and made revision accordingly.** The detailed responses are listed as follows:

**[Q1]** More clear motivation.

**[A1]** Thank you for pointing this out. We clarify that the works [5, 54] were primarily intended to illustrate the existence of complex, non-deterministic image transformations performed by OSNs, which disrupt pixel-level perturbations. Furthermore, references [23, 33, 44, 46] all include relevant descriptions indicating that the perturbations caused by OSN transmission will affect the effectiveness of adversarial examples. We agree that the citation context was ambiguous and will reshape the phrasing for clarity enhancement. Our key insight is that these uncontrollable transformations can be leveraged as part of the attack process. Rather than resisting OSN degradation (as robust AEs aim to do), we propose a novel paradigm, Stealthy-AE, where the adversarial property is intentionally hidden before OSN transmission and is triggered by the OSN's inherent processing. This reframes OSNs not as obstacles, but as vectors to carry hidden adversarial intent, which is fundamentally different from prior AE assumptions. We thank the reviewer for your suggestions, which will help us improve the clarity and quality of the manuscript presentation.

**[Q2]** The necessity of using the denoising diffusion layer.

**[A2]** We appreciate the reviewer’s useful comments and suggestions. To validate the design of our simulated OSN network, we conducted a detailed ablation study evaluating the role of the denoising diffusion layer as well as other key components. The results (with Facebook, against ResNet by FGSM) are summarized in the table below. Removing the denoising diffusion layer causes a 17.51% drop in OSN-AE success rate and increased instability (higher Both-AE/Non-AE cases), indicating that the denoising layer improves the fidelity of transformation modeling. Replacing diffusion with a shallow U-Net baseline results in weaker generalization and an 11.94% performance gap. Removing the differentiable JPEG layer causes the most severe degradation, highlighting the importance of joint modeling. When bypassing the OSN simulation entirely and directly applying FGSM + JPEG, the model fails to produce reliable OSN-AEs, confirming that simulation fidelity is essential. These results confirm that the denoising diffusion layer significantly contributes to the robustness and precision of OSN-AE generation by approximating OSN operations in Stealthy-AE designs. We have supplemented these experimental statements and explanations this ablation in the final revision.

| Configuration                             | OSN-AE (%) | Vanilla-AE (%) | Both-AE (%) | Non-AE (%) |
| ----------------------------------------- | ---------- | -------------- | ----------- | ---------- |
| Full model (ours)                         | **94.24**  | 0.00           | 2.27        | 3.49       |
| w/o Denoising Diffusion Layer             | 76.73      | 0.00           | 10.35       | 12.92      |
| w/o Differentiable JPEG Layer             | 69.81      | 0.00           | 12.74       | 17.45      |
| Replace Diffusion with U-Net Encoder Only | 82.30      | 0.00           | 7.23        | 10.47      |
| w/o OSN Simulation (Direct FGSM + JPEG)   | 41.12      | 10.15          | 18.68       | 30.14      |


**[Q3]** About method for generating adversarial examples.

**[A3]** Thank the reviewer for your comment. Our main novelty lies in introducing a new attack paradigm that leverages OSN transmission as a stealthy adversarial channel. To the best of our knowledge, this is the first work to model non-adversarial-to-adversarial transitions induced by real-world platform transformations. In fact, our adversarial example generation method is also born out of the above core ideas. We formalize the AE generation and extend it using Lagrangian optimization to enforce the preconditions and postconditions required for the definition of stealthy AE (as introduced in Section 3.4). In the manuscript, we integrate 9 representative AE generation methods. Due to space limitations, we cannot present all the formulas for combining the 9 attacks with Lagrangian optimization in the manuscript, we will put these formulas in an online anonymous repository. Overall, as shown in Table 4, and the results show the wide applicability of our Lagrangian optimization method in Stealthy-AE with various AE generations. 

## Dear Reviewer G2fZ:
*Rating (5); Technical Quality (7); Presentation Quality (6)*

Thank you for your very detailed and constructive comments and suggestions on our manuscript. Sorry for the late reply, given we supplemented some experiments responding to your comments. **We have studied the comments carefully and made revision accordingly.** The detailed responses are listed as follows:

**[Q1]** About deep insights into effectiveness and the empirical ablation study.

**[A1]** Formally, we model OSN transformations as a learnable composition of (i) structured degradation (JPEG compression) and (ii) unstructured, content-dependent distortions (e.g., enhancement filters or latent reconstruction), which are well approximated via denoising diffusion networks. This formal modeling is consistent with the line of previous work [23, 33, 44, 46], which show OSN pipelines involve complex and data-dependent operations beyond compression. More importantly, our modeling choice is **not merely heuristic**, but is embedded into a constrained optimization formulation. Specifically, in Sections 3.1 and 3.4, we define the stealthy AE generation task as the following:

Find δ such that the perturbed image $x' = x + \delta$ satisfies:
$$
f(x') = f(x) \quad \text{(correct before transmission)}
$$

$$
f(T(x')) = y_{\text{target}} \quad \text{(misclassified after simulated OSN)}
$$

$$
\|\delta\| \leq \epsilon \quad \text{(imperceptibility constraint)}
$$

Here, $T(x')$ is our **OSN simulation**, defined as $T(x') = \text{Diffusion}(\text{JPEG}(x'))$. This formulation induces a **Lagrangian optimization problem**:

$$
\mathcal{L}(\delta, \lambda_1, \lambda_2) = L_{attack}(f(T(x + \delta)), y_{target}) + \lambda_1 (L(f(x + \delta), y) - \epsilon_{class}) + \lambda_2 (d(x, x + \delta) - \epsilon)
$$

This design encourages the optimization process to **explicitly align the adversarial objective with the OSN simulation**, meaning that the generated perturbation is *not only stealthy* but also **optimized with respect to the simulated degradation pipeline**. Moreover, by backpropagating through both the diffusion and JPEG layers, we allow the attack to explore perturbation directions that are robust to structured-unstructured distortions jointly. In summary, the theoretical support lies in: (i) A two-level decomposition of OSN transformations into structured + unstructured components; (ii) A differentiable, end-to-end simulation that enables gradient-based adversarial optimization; (iii) An explicit constrained optimization framework that guides the solution toward "clean-before / adversarial-after" objectives, i.e., true stealthy AEs.

**Empirically**, to validate the design of our simulated OSN network, we conducted a detailed ablation study evaluating the role of the denoising diffusion layer as well as other key components. The results (with Facebook, against ResNet by FGSM) are summarized in the table below. Removing the denoising diffusion layer causes a 17.51% drop in OSN-AE success rate and increased instability (higher Both-AE/Non-AE cases), indicating that the denoising layer improves the fidelity of transformation modeling. Replacing diffusion with a shallow U-Net baseline results in weaker generalization and an 11.94% performance gap. Removing the differentiable JPEG layer causes the most severe degradation, highlighting the importance of joint modeling. When bypassing the OSN simulation entirely and directly applying FGSM + JPEG, the model fails to produce reliable OSN-AEs, confirming that simulation fidelity is essential. These results confirm that the denoising diffusion layer significantly contributes to the robustness and precision of OSN-AE generation by approximating OSN operations in Stealthy-AE designs. We have supplemented these experimental statements and explanations this ablation in the final revision.

| Configuration                             | OSN-AE (%) | Vanilla-AE (%) | Both-AE (%) | Non-AE (%) |
| ----------------------------------------- | ---------- | -------------- | ----------- | ---------- |
| Full model (ours)                         | **94.24**  | 0.00           | 2.27        | 3.49       |
| w/o Denoising Diffusion Layer             | 76.73      | 0.00           | 10.35       | 12.92      |
| w/o Differentiable JPEG Layer             | 69.81      | 0.00           | 12.74       | 17.45      |
| Replace Diffusion with U-Net Encoder Only | 82.30      | 0.00           | 7.23        | 10.47      |
| w/o OSN Simulation (Direct FGSM + JPEG)   | 41.12      | 10.15          | 18.68       | 30.14      |

**[Q2]** Real-world OSN validation.

**[A2]** Thank you for your thoughtful comment. In fact, our OSN simulation model is trained and validated using data derived from real-world platforms. As stated in Section 4.1.1, our training and evaluation datasets are based on the benchmark provided by [23], which includes 3000 original ImageNet images and their post-transmission counterparts from three real-world OSNs: Facebook, QQ, and WeChat. These images were collected by uploading and retrieving samples through the actual platforms, ensuring that our simulated model learns transformations from real OSN behavior.

To further address your concern, we conducted an additional real-world validation of the adversarial effectiveness. Specifically, we generate OSN-AEs using ResNet as the victim model, and the three representative attacks, FGSM, PGD, and C&W. Then, we upload these generated adversarial samples to the actual OSNs (Facebook, QQ, and WeChat), retrieve the compressed results, and test whether they successfully fool the ResNet model. The results are summarized in the table below. These results confirm that the adversarial properties produced by our Stealthy-AE framework persist under real OSN transformations, with only slight drops (mostly within 1–2%) compared to simulated results. This demonstrates the high fidelity of our OSN simulation model, and validates that the generated OSN-AEs are indeed effective in real-world scenarios. We will include this table and discussion in the final version to explicitly address the real-world validation aspect.

| OSN Platform | Attack Method | Simulated OSN-AE Success Rate (Table 2) | **Real OSN-AE Success Rate (Validation)** |
| ------------ | ------------- | --------------------------------------- | ----------------------------------------- |
| Facebook     | FGSM | 94.24% | 92.17% |
| Facebook     | PGD | 86.11% | 84.02% |
| Facebook     | C\&W | 94.93% | 92.85% |
| QQ           | FGSM | 97.90% | 96.81% |
| QQ           | PGD | 91.88% | 90.46% |
| QQ           | C\&W | 95.37% | 94.19% |
| WeChat       | FGSM | 97.47% | 96.30% |
| WeChat       | PGD | 99.65% | 98.71% |
| WeChat       | C\&W | 95.28% | 94.80% |


**[Q3]** Missing training details.

**[A3]** Thank you for your suggestion. To improve reproducibility, we provide additional details on the training of our simulated OSN model. As described in Section 4.1.1, we use the benchmark dataset from [23], which includes 3,000 original ImageNet images and their real-world transmitted versions through Facebook, QQ, and WeChat. This results in 9,000 OSN-transformed images paired with the originals, yielding 12,000 total samples with a 1:1 training and testing split. We train the model for 200 epochs using the Adam optimizer with an initial learning rate of 1e-4 (cosine annealing), batch size of 16, and L2 loss between the simulated and real OSN outputs as in Eq. (7). The Autoencoder-KL, Autodecoder-KL, and CLIP encoder are frozen during training; only the residual and attention blocks of the diffusion module are updated. The training converges after around 100 epochs, with validation PSNR consistently above 40dB and SSIM exceeding 0.91, as reported in Section 4.2. The total training time is around 6 hours on an RTX 3090 GPU. We incorporated these details in the final vision. 

**[Q4]** About applicability.

**[A4]** Thank you for the suggestion. While our current experiments focus on classification (ResNet, VGG, Inception), we emphasize that Stealthy-AE is modular and orthogonal to the task type. The simulated OSN module operates directly on images and is agnostic to the downstream model. This is consistent with most previous work focusing on adversarial examples, such as [MM_Jia, MM_Ge], which only experiments on image classification models. Nevertheless, we are willing to explore the results of Stealthy-AE’s OSN-AE idea in tasks such as object detection and image segmentation in the future, and we will also supplement these contents in the discussion section. 

[MM_Jia] Jia X, Wei X, Cao X, et al. Adv-watermark: A novel watermark perturbation for adversarial examples. ACM MM 2020.
[MM_Ge] Ge Z, Shang F, Liu H, et al. Improving the transferability of adversarial examples with arbitrary style transfer. ACM MM 2023.

**[Q5]** About baseline analysis.

**[A5]** Thank you for your insightful comment. Beyond reporting success rates, we conduct multiple in-depth analyses throughout Secions 4.3-4.5, to understand why Stealthy-AE consistently outperforms existing baselines. First, in Section 4.3 we compare across different OSNs and victim models, revealing that Stealthy-AE performs robustly even under challenging conditions such as low-quality compression (e.g., WeChat with QF=58) and resilient architectures. In Section 4.4, we evaluate resistance to seven state-of-the-art AE detectors, showing that Stealthy-AE achieves the lowest detection rates across the board (Table 3), indicating superior stealthiness and robustness. Furthermore, in Section 4.5.1 we analyze quality metrics such as PSNR and SSIM of the generated OSN-AEs, demonstrating that Stealthy-AE achieves high visual fidelity (PSNR > 40dB) while still maintaining attack effectiveness. In Section 4.5.2–4.5.3, we explore performance under varying perturbation norms ($\ell_2$ and $\ell_\infty$), attack types (targeted vs. untargeted), and across 8 different victim models, including robustly trained ones like TRADES and YOPO. These extensive studies highlight that our advantage is not limited to raw success rate: rather, Stealthy-AE excels because it (i) explicitly optimizes for the OSN transmission process, (ii) preserves stealth and quality, and (iii) generalizes across platforms, models, and attacks. We will revise the paper to better emphasize these analytical findings and explicitly clarify why Stealthy-AE is effective.

**[Q6]** About Generality.

**[A6]** We appreciate the reviewer’s feedback on our generality claim. First, we clarify a potential misunderstanding: while FGSM, PGD, C&W, and MIFGSM are emphasized in the main evaluations, Table 4 in Section 4.5.3 demonstrates the compatibility of Stealthy-AE with 9 diverse adversarial attack methods (FGSM, PGD, BIM, DeepFool, C&W, DIM, NES, SPSA, NATTACK), including not only gradient-based attacks, but also black-box attacks and query-efficient methods, evaluated on eight victim models including robust architectures like TRADES and YOPO. This empirical evidence supports that our framework is indeed attack-agnostic and adaptable across different threat models. Regarding the specific attack types mentioned in [1-3], we acknowledge their relevance. While we did not include AutoAttack [1] and recent diffusion-based generative attacks [2,3] in the current version due to evaluation overhead and reproducibility complexity, our design is inherently compatible with them. In fact, our Lagrangian optimization framework (Section 3.4) is modular and can incorporate any differentiable or query-based loss, as long as it produces adversarial gradients or objectives. Notably, for [2,3], which use conditional/generative diffusion models, we believe their integration could be symbiotic with our OSN-simulation diffusion module, potentially forming a dual-diffusion architecture. For [1], AutoAttack ensembles multiple attacks (APGD-CE, APGD-DLR, Square, FAB), most of which are compatible as subroutines within our optimization pipeline. We will highlight that Stealthy-AE has been integrated with 9 adversarial attacks and discuss future considerations for integrating more attacks.

**[Q7]** Consider dynamic JPEG quality factor or extreme compression.

**[A7]** Thank you for your insightful suggestions. We agree with your point of view that image data could involve different transformations when transmitted on different OSN platforms. As we introduced in Section 3.2 and Figure 3 of the manuscript, Facebook, QQ, and WeChat have different Quality Factors (QF) for JPEG image compression. However, on the one hand, the calculation of the QF does not involve reverse-engineering operations, and we can calculate the QF based on the standard quantization tables for luminance (table 0) and chrominance (table 1) defined by the JPEG standard (ISO/IEC 10918-1) (the specific matrix values ​​can be found in the online anonymous github repository). On the other hand, we acknowledge that different OSN platforms may have more complex operation processes, and we do not oppose this scenario (we will add the discussion of considering more platforms in the final version), so in the manuscript we separate OSN to facilitate horizontal expansion of the simulation of more Online Social Networks. Nevertheless, we additionally set the OSN network to a more extreme QF value (and set QF to a random dynamic strategy between [0, 100]) to explore the effect of Stealthy-AE when the simulated OSN and the real test network have different QF values. The results (as the Table below) show that despite the setting of extreme QF values, Stealthy-AE still generates more than 90% of OSN-AE (significantly outperforms other baselines in Figure 5), which demonstrates the practicality and stability of our proposal.

| OSN      | Setting | OSN-AE (%) | Vanilla-AE (%) | Both-AE (%) | Non-AE (%) |
| -------- | ---------- | ---------- | -------------- | ----------- | ---------- |
| Facebook | QF=50      | 93.01      | 0.00           | 3.12        | 3.97       |
| Facebook | QF=30      | 91.42      | 0.00           | 3.85        | 4.73       |
| Facebook | QF=10      | 90.16      | 0.00           | 4.51        | 5.33       |
| Facebook | QF=Dynamic | 93.62      | 0.00           | 2.79        | 3.59       |
| QQ       | QF=50      | 96.54      | 0.00           | 1.02        | 2.44       |
| QQ       | QF=30      | 94.76      | 0.00           | 2.66        | 3.58       |
| QQ       | QF=10      | 91.35      | 0.00           | 3.23        | 5.42       |
| QQ       | QF=Dynamic | 96.93      | 0.00           | 1.35        | 1.72       |
| WeChat   | QF=50      | 97.06      | 0.00           | 0.54        | 2.40       |
| WeChat   | QF=30      | 94.23      | 0.00           | 1.16        | 4.61       |
| WeChat   | QF=10      | 92.91      | 0.00           | 2.20        | 4.89       |
| WeChat   | QF=Dynamic | 97.12      | 0.00           | 1.36        | 1.52       |

**[Q8]** About anonymity.

**[A8]** We strictly abide by the anonymity rules, including online GitHub repositories, GitHub account names, repository names, and upload ID names are all anonymous, and no author information can be identified. Thank you for your kind reminder.

**[Q9]** Consider more platforms

**[A9]** Thank you for the insightful question. While our current implementation targets JPEG-based pipelines (common in Facebook, QQ, and WeChat), our framework is not inherently limited to JPEG compression. As described in Section 3.3, the compression module is implemented as a differentiable transformation layer, and can be replaced or extended to WebP or other codecs (e.g., AVIF, proprietary formats). For platforms like TikTok, we plan to train a new simulation module using upload-download image pairs, similar to how we used real OSN-transformed data from [23]. If a differentiable implementation of the target compression codec is not available, it can be approximated via supervised training of a neural image-to-image model, as we did for the unstructured distortion component (diffusion layer). We will mention this extensibility in the final version.

**[Q10]** About detection robustness

**[A10]** We thank the reviewer for raising this concern. In Secion 4.4 and Table 3, we evaluate Stealthy-AE against seven advanced AE detection methods, including robustness-aware detectors such as PRest-Net [5] and IFD [46], which explicitly account for OSN-specific transformations. Our method achieves the lowest detection rates across all detectors, with PRest-Net at 38.6% and IFD at 40.1%, significantly outperforming strong baselines like SIO and ComModel. These results suggest that our OSN-AEs preserve stealth not only visually but also in the latent feature space, making them difficult to detect even for OSN-aware and ensemble-based strategies. We will highlight this more clearly in the final vision.

**[Q11]** Perceptual quality assessment

**[A11]** Thank you for the suggestion. While we did not conduct a formal user study, we do include objective perceptual quality metrics in Section 4.5.1: PSNR and SSIM are used to evaluate the fidelity of generated OSN-AEs. As shown in Figure 6(b), the PSNR values of Stealthy-AEs consistently exceed 40dB, and SSIM remains above 0.91, which are generally considered perceptually indistinguishable from the original images. These results are in line with prior work [23, 33] and support that our perturbations are imperceptible. We acknowledge that a human evaluation could offer complementary insights, and we plan to include one in future work. In addition, the Stealthy AE (bottom) in Figure 1 is also an example we generated. It can be seen that the image does not have obvious perceptually noticeable artifacts or degraded semantic content.

**[Q12]** Lagrange optimization parameters.

**[A12]** We appreciate the reviewer’s interest in the optimization formulation. The **Lagrange multipliers $\lambda_1$ and $\lambda_2$** in Eq. (9) are treated as tunable hyperparameters. In our experiments, we conduct a lightweight grid search to find a balance between stealth (small $\lambda_2$) and effectiveness (large $\lambda_1$). We find the method is **relatively robust** to moderate changes in these values. Specifically, for all attacks evaluated (FGSM, PGD, C\&W), we use the same default setting $\lambda_1 = 1.0$, $\lambda_2 = 0.5$, which achieves high OSN-AE success rates and imperceptibility across OSNs. To evaluate the sensitivity of Stealthy-AE to Lagrange multiplier settings, we conduct an ablation study on the WeChat platform using the Inception model, under three representative attacks (FGSM, PGD, C\&W). We vary $\lambda_1$ (controls pre-OSN classification correctness) and $\lambda_2$ (controls perturbation budget) within a reasonable range and report the OSN-AE success rate (%). The default setting in the main paper is $\lambda_1 = 1.0$, $\lambda_2 = 0.5$. As shown in the table, the performance of Stealthy-AE remains **stable across different Lagrange settings**, with success rate fluctuations within $\pm$2%. Minor decreases of OSN-AE with low $\lambda_1$ (weaker constraint on pre-OSN correctness). Overall, the results demonstrate that Stealthy-AE is **not overly sensitive to the choice of multipliers**.

| Attack | $\lambda_1$ = 0.5, $\lambda_2$ = 0.5 | $\lambda_1$ = 1.0, $\lambda_2$ = 0.5 (default) | $\lambda_1$ = 2.0, $\lambda_2$ = 0.5 |
|--------|---------------------|------------------------------|--------------------|
| FGSM   | 93.45%              | 94.20%                       | 96.17%             |
| PGD    | 96.81%              | 97.79%                       | 97.96%             |
| C&W    | 94.88%              | 95.12%                       | 96.74%             |

**[Q12]** Other minors.

**[A12]** Thanks for your constructive comments about Figure and Table clarity, typos, notation and formatting, visualization, and conclusion statements. We have revised the issues accordingly.

## Dear Reviewer S97i:
*Rating (5); Technical Quality (5); Presentation Quality (6)*

**[Q1]** About novelty and method for generating adversarial examples.

**[A1]** Thank the reviewer for your comment. Our main novelty lies in introducing a new attack paradigm that leverages OSN transmission as a stealthy adversarial channel. To the best of our knowledge, this is the first work to model non-adversarial-to-adversarial transitions induced by real-world platform transformations. In fact, our adversarial example generation method is also born out of the above core ideas. We formalize the AE generation and extend it using Lagrangian optimization to enforce the preconditions and postconditions required for the definition of stealthy AE (as introduced in Section 3.4). In the manuscript, we integrate 9 representative AE generation methods. Due to space limitations, we cannot present all the formulas for combining the 9 attacks with Lagrangian optimization in the manuscript, we will put these formulas in an online anonymous repository. Overall, as shown in Table 4, and the results show the wide applicability of our Lagrangian optimization method in Stealthy-AE with various AE generations. 

**[Q2]** About the clarification of Tables.

**[A2]** In Table 3, all AE detection results are tested based on all generated OSN-AEs. In other words, the results presented in Table 3 are the average results of the three attacks, i.e., FGSM, PGD, and C&W. Moreover, Table 4 in Section 4.5.3 demonstrates the compatibility of Stealthy-AE with 9 diverse adversarial attack methods (FGSM, PGD, BIM, DeepFool, C&W, DIM, NES, SPSA, NATTACK), including not only gradient-based attacks, but also black-box attacks and query-efficient methods, evaluated on eight victim models including robust architectures like TRADES and YOPO. This empirical evidence supports that our framework is attack-agnostic and adaptable across different threat models. 

**[Q3]** Analysis of Stealthy-AE effectiveness.

**[A3]** Formally, we model OSN transformations as a learnable composition of (i) structured degradation (JPEG compression) and (ii) unstructured, content-dependent distortions (e.g., enhancement filters or latent reconstruction), which are well approximated via denoising diffusion networks. This formal modeling is consistent with the line of previous work [23, 33, 44, 46], which show OSN pipelines involve complex and data-dependent operations beyond compression. More importantly, our modeling choice is **not merely heuristic**, but is embedded into a constrained optimization formulation. Specifically, in Sections 3.1 and 3.4, we define the stealthy AE generation task as the following:

Find δ such that the perturbed image $x' = x + \delta$ satisfies:
$$
f(x') = f(x) \quad \text{(correct before transmission)}
$$

$$
f(T(x')) = y_{\text{target}} \quad \text{(misclassified after simulated OSN)}
$$

$$
\|\delta\| \leq \epsilon \quad \text{(imperceptibility constraint)}
$$

Here, $T(x')$ is our **OSN simulation**, defined as $T(x') = \text{Diffusion}(\text{JPEG}(x'))$. This formulation induces a **Lagrangian optimization problem**:

$$
\mathcal{L}(\delta, \lambda_1, \lambda_2) = L_{attack}(f(T(x + \delta)), y_{target}) + \lambda_1 (L(f(x + \delta), y) - \epsilon_{class}) + \lambda_2 (d(x, x + \delta) - \epsilon)
$$

This design encourages the optimization process to **explicitly align the adversarial objective with the OSN simulation**, meaning that the generated perturbation is *not only stealthy* but also **optimized with respect to the simulated degradation pipeline**. Moreover, by backpropagating through both the diffusion and JPEG layers, we allow the attack to explore perturbation directions that are robust to structured-unstructured distortions jointly. In summary, the theoretical support lies in: (i) A two-level decomposition of OSN transformations into structured + unstructured components; (ii) A differentiable, end-to-end simulation that enables gradient-based adversarial optimization; (iii) An explicit constrained optimization framework that guides the solution toward "clean-before / adversarial-after" objectives, i.e., true stealthy AEs.

**Empirically**, to validate the design of our simulated OSN network, we conducted a detailed ablation study evaluating the role of the denoising diffusion layer as well as other key components. The results (with Facebook, against ResNet by FGSM) are summarized in the table below. Removing the denoising diffusion layer causes a 17.51% drop in OSN-AE success rate and increased instability (higher Both-AE/Non-AE cases), indicating that the denoising layer improves the fidelity of transformation modeling. Replacing diffusion with a shallow U-Net baseline results in weaker generalization and an 11.94% performance gap. Removing the differentiable JPEG layer causes the most severe degradation, highlighting the importance of joint modeling. When bypassing the OSN simulation entirely and directly applying FGSM + JPEG, the model fails to produce reliable OSN-AEs, confirming that simulation fidelity is essential. These results confirm that the denoising diffusion layer significantly contributes to the robustness and precision of OSN-AE generation by approximating OSN operations in Stealthy-AE designs. We have supplemented these experimental statements and explanations this ablation in the final revision.

| Configuration                             | OSN-AE (%) | Vanilla-AE (%) | Both-AE (%) | Non-AE (%) |
| ----------------------------------------- | ---------- | -------------- | ----------- | ---------- |
| Full model (ours)                         | **94.24**  | 0.00           | 2.27        | 3.49       |
| w/o Denoising Diffusion Layer             | 76.73      | 0.00           | 10.35       | 12.92      |
| w/o Differentiable JPEG Layer             | 69.81      | 0.00           | 12.74       | 17.45      |
| Replace Diffusion with U-Net Encoder Only | 82.30      | 0.00           | 7.23        | 10.47      |
| w/o OSN Simulation (Direct FGSM + JPEG)   | 41.12      | 10.15          | 18.68       | 30.14      |

Beyond reporting success rates, we conduct multiple in-depth analyses throughout Secions 4.3-4.5, to understand why Stealthy-AE consistently outperforms existing baselines. First, in Section 4.3 we compare across different OSNs and victim models, revealing that Stealthy-AE performs robustly even under challenging conditions such as low-quality compression (e.g., WeChat with QF=58) and resilient architectures. In Section 4.4, we evaluate resistance to seven state-of-the-art AE detectors, showing that Stealthy-AE achieves the lowest detection rates across the board (Table 3), indicating superior stealthiness and robustness. Furthermore, in Section 4.5.1 we analyze quality metrics such as PSNR and SSIM of the generated OSN-AEs, demonstrating that Stealthy-AE achieves high visual fidelity (PSNR > 40dB) while still maintaining attack effectiveness. In Section 4.5.2–4.5.3, we explore performance under varying perturbation norms ($\ell_2$ and $\ell_\infty$), attack types (targeted vs. untargeted), and across 8 different victim models, including robustly trained ones like TRADES and YOPO. These extensive studies highlight that our advantage is not limited to raw success rate: rather, Stealthy-AE excels because it (i) explicitly optimizes for the OSN transmission process, (ii) preserves stealth and quality, and (iii) generalizes across platforms, models, and attacks. We will revise the paper to better emphasize these analytical findings and explicitly clarify why Stealthy-AE is effective.

## Dear Reviewer Uet7:
*Rating (7); Technical Quality (7); Presentation Quality (7)*

**[Q1]** About overhead.

**[A1]** Thanks for your constructive comments on the computational cost of the diffusion model and its applicability in real-time or time-constrained adversarial scenarios. To address this concern, we measured the average inference time and memory overhead per image (batch size = 1) across all methods on the same GPU setup (NVIDIA RTX 3090). The results are summarized below:

| **Method**           | **Inference Time (s)** | **Memory Footprint (MB)** | **Additional Overhead**     | **Remarks**                                  |
| -------------------- | ---------------------- | ------------------------- | --------------------------- | -------------------------------------------- |
| Base (FGSM)          | 0.03                   | 300                       | None                        | Standard FGSM/PGD implementation             |
| JPEGR \[38]          | 0.10                   | 320                       | JPEG compression simulation | Simple JPEG encoder model                    |
| UAP \[29]            | 0.07                   | 330                       | Pretrained universal noise  | Offline training needed                      |
| FIA \[52]            | 0.25                   | 380                       | Fourier transform           | Adds FFT memory overhead                     |
| ComModel \[33]       | 0.21                   | 450                       | Compressed feature encoder  | Dual-path feature pipeline                   |
| ComGAN \[44]         | 0.51                   | 580                       | GAN inference engine        | Multi-module architecture                    |
| SIO \[23]            | 0.45                   | 500                       | OSN simulation module       | Multi-stage pipeline                         |
| **Ours (Diffusion)** | **0.47**               | **480**                   | Fast diffusion (e.g., DDIM) | Efficient, end-to-end, OSN-robust simulation |

The results indicate that the diffusion model indeed our method indeed introduces additional computational cost, a slightly longer and acceptable inference time (0.47s). Even though, it maintains a **moderate memory footprint (480MB)**, which is comparable to ComModel [33] and **lower than ComGAN [44] and SIO [23]**. This can be attributed to the fact that they rely on complex multi-stage architectures or adversarial simulation modules. This demonstrates that our approach is **not inherently more resource-intensive in terms of space**. In contrast, the diffusion network can proactively model the transformation behavior of OSNs, enabling our model to simulate and adapt to OSN-induced perturbations in advance and produce more robust adversarial examples. Furthermore, while diffusion models are often viewed as computationally heavy, their footprint is largely determined by the number of sampling steps, which can be significantly reduced using techniques like **DDIM, FastDiff, or distillation-based approximations**. We plan to incorporate such strategies in future optimizations and discuss them in the paper. Overall, our method offers a **strong trade-off between time, space, and robustness**, making it highly practical for OSN-facing adversarial applications.

**[Q2]** About the scalability.

**[A2]** Thank you for this valuable suggestion. Our framework is designed with scalability in mind. First, our diffusion-based OSN simulator operates in a latent space, following the autoencoder-based formulation in Section 3.3. This significantly reduces computational overhead compared to pixel-space diffusion models, allowing the framework to scale efficiently to higher-resolution images (e.g., 512×512 or 1024×1024) by simply adjusting the encoder/decoder resolution without increasing the number of diffusion steps. Second, our Stealthy-AE optimization is modular and attack-agnostic, meaning it can be parallelized across larger datasets or multiple GPUs with minimal changes. 

Furthermore, Table 4 in Section 4.5.3 demonstrates the compatibility of Stealthy-AE with 9 diverse adversarial attack methods (FGSM, PGD, BIM, DeepFool, C&W, DIM, NES, SPSA, NATTACK), including not only gradient-based attacks, but also black-box attacks and query-efficient methods, evaluated on eight victim models including robust architectures like TRADES and YOPO. This empirical evidence supports that our framework is indeed attack-agnostic and adaptable across different threat models. We will highlight that Stealthy-AE has been integrated with 9 adversarial attacks and discuss future considerations for integrating more attacks.


