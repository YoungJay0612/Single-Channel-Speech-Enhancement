# Single Channel Speech Enhancement
记录关于单通道语音增强，包括降噪和去混响等相关的论文，博客和代码等。

## GuideBoook
github链接：https://github.com/WenzheLiu-Speech/The-guidebook-of-speech-enhancement  


## DNS比赛
github地址： https://github.com/microsoft/DNS-Challenge  
网页地址：https://www.microsoft.com/en-us/research/academic-program/deep-noise-suppression-challenge-icassp-2022/  

## Speech Deep Learning
### 2024-7-17
Beyond Performance Plateaus: A Comprehensive Study on Scalability in Speech Enhancement  
标题：超越性能瓶颈：语音增强可扩展性全面研究  
论文作者：张王优，Kohei Saijo，Jee-weon Jung，李晨达，Shinji Watanabe，钱彦旻  
完成单位：上海交通大学 ，早稻田大学，卡耐基梅隆大学  
链接：https://arxiv.org/pdf/2406.04269  
论文亮点：该论文深入探究了不同架构的语音增强模型（BSRNN、Conv-TasNet、DEMUCS-v4、TF-GridNet）在不同模型复杂度、不同训练数据量、因果/非因果模式等情况下的规模化能力。相关实验采用了来自多个领域的公开数据（VCTK+DEMAND、DNS-2020、WHAMR!、CHiME-4、REVERB），以评估模型的泛化能力和处理不同声学环境的通用性。实验结果揭示了不同模型架构在语音增强规模化能力上的显著差别，也指出了语音增强领域当前尚待探索的研究方向，如构建大规模多领域数据集和设计可高效规模化的模型架构。  

URGENT Challenge: Universality, Robustness, and Generalizability For Speech Enhancement  
标题：URGENT挑战赛：语音增强的通用性、鲁棒性和泛化性  
论文作者：张王优,Robin Scheibler，Kohei Saijo，Samuele Cornell，李晨达，倪兆衡，Anurag Kumar，Jan Pirklbauer，Marvin Sach6，Shinji Watanabe，Tim Fingscheidt，钱彦旻  
完成单位：上海交通大学，卡耐基梅隆大学 ，LY Corporation，早稻田大学，Meta，布伦瑞克工业大学  
论文亮点：该论文介绍了一项全新的语音增强比赛——URGENT 2024，其着重关注语音增强模型的通用性、鲁棒性和泛化性，旨在构建单个通用模型来同时处理具有不同失真和不同采样率的语音信号。区别于大部分现有语音增强比赛，URGENT 2024采用了更广义的语音增强任务定义（包含多种子任务）、大规模多领域数据（且约束了可用训练数据以保证公平性）以及十分多样的评估指标（四大类）。该论文还介绍了针对所提出的全新任务的通用语音增强框架，并基于多种常见语音增强模型进行了验证性实验。  
链接：https://arxiv.org/abs/2406.04660  

### 2022-11-2
Parallel Gated Neural Network With Attention Mechanisim For Speech Enhancement  
标题：具有注意机制的并行门控神经网络语音增强  
链接：https://arxiv.org/abs/2210.14509  
作者：Jianqiao Cui,Stefan Bleeck  
机构：Institute of Sound and Vibration Research, University of Southampton, UK  
备注：5 pages, 6 figures, references added  

### 2022-7-1
paper title:  Comparing Conventional Pitch Detection Algorithms with a Neural Network Approach  
标题：传统基音检测算法与神经网络方法的比较  
链接：https://arxiv.org/abs/2206.14357  
作者：Anja Kroon  
机构：ECSE , Speech Communications Final Project, Dept. of Electrical and Computer Engineering, McGill University, Montreal, Quebec, Canada  
备注：6 pages, 11 figures  

### 2022-4-12
paper title: DDOS: A MOS Prediction Framework utilizing Domain Adaptive Pre-training and Distribution of Opinion Scores  
标题：DDoS：一种基于领域自适应预训练和意见得分分布的MOS预测框架  
作者：Wei-Cheng Tseng,Wei-Tsung Kao,Hung-yi Lee  
机构*：Graduate Institute of Communication Engineering, National Taiwan University  
备注：Submitted to Interspeech 2022. Code will be available in the future  
链接：https://arxiv.org/pdf/2204.03219v1.pdf  


paper title: FFC-SE: Fast Fourier Convolution for Speech Enhancement  
标题：FFC-SE：用于语音增强的快速傅立叶卷积  
作者：Ivan Shchekotov,Pavel Andreev,Oleg Ivanov,Aibek Alanov,Dmitry Vetrov  
机构*：∗ Equal contribution, Samsung AI Center, Moscow, Higher School of Economics, Moscow, Skolkovo Institute of Science and Technology, Moscow, Artificial Intelligence Research Institute, Moscow  
备注：https://arxiv.org/pdf/2204.03042v1.pdf  


### 2021-12-15
paper title: Shennong: a Python toolbox for audio speech features extraction  
标题：神农：一个用于音频语音特征提取的Python工具箱  
链接：https://arxiv.org/abs/2112.05555  
作者：Mathieu Bernard,Maxime Poli,Julien Karadayi,Emmanuel Dupoux  
机构：Dupoux, Received: date Accepted: date  

### 2021-11-22
paper title: A Conformer-based ASR Frontend for Joint Acoustic Echo Cancellation, Speech Enhancement and Speech Separation  
标题：一种基于Conform的声学回波抵消、语音增强和语音分离联合ASR前端  
链接：https://arxiv.org/abs/2111.09935  
作者：Tom O'Malley,Arun Narayanan,Quan Wang,Alex Park,James Walker,Nathan Howard  
机构：Google LLC, U.S.A  
备注：Will appear in IEEE-ASRU 2021  

### 2021-10-19
paper:NN3A: Neural Network supported Acoustic Echo Cancellation, Noise Suppression and Automatic Gain Control for Real-Time Communications  
标题：NN3A：神经网络支持的声学回波抵消、噪声抑制和自动增益控制的实时通信  
链接：https://arxiv.org/abs/2110.08437  
作者：Ziteng Wang,Yueyue Na,Biao Tian,Qiang Fu  
机构：Alibaba Group, China  
备注：submitted to ICASSP2022  

### 2021-8-20
paper:Continuous Speech Separation with Conformers  ICASSP2021  
github：https://github.com/Sanyuan-Chen/CSS_with_Conformer  
paper link: https://arxiv.org/abs/2008.05773  


### 2021-6-5
Wav2Vec2.0:  
model links: https://github.com/pytorch/fairseq/tree/master/examples/wav2vec  
using links: https://github.com/awslabs/speech-representations  
paper links: https://arxiv.org/abs/2006.11477  


### 2021-4-23
paper title: Interpreting intermediate convolutional layers of CNNs trained on raw speech 解释在原始语音上训练的CNN的中间卷积层  
paper link: https://arxiv.org/pdf/2104.09489.pdf  


## Speech Denoise

### 2023-6-26
paper title: D2Former: a Fully Complex Dual-Path Dual-Decoder Conformer Network Using Joint Complex Masking and Complex Spectral Mapping for Monaural Speech Enhancement  
链接：https://arxiv.org/abs/2302.11832  
作者：阿里巴巴  
备注：ICASSP2023  
描述：提出了一个基于Conformer结构的完全使用复数值网络的语音增强模型：D2Former。在D2Former设计中，我们将Conformer中的实数值注意力机制扩展到复数值注意力机制，并结合时间序列和频域序列的双路径处理模式，更有效地对复数值时频语音特征序列进行建模。我们基于沿时间轴的复数值扩张卷积（Dilation Convolution）和沿频率轴的递归复数值前馈序列记忆网络 (Complex FSMN)，通过双路径学习模式进一步提升编码器和解码器中的时频特征表示和处理能力。此外，我们通过一个多任务联合学习框架来结合复数值掩蔽和复数值频谱映射两个训练目标的优势，来改善模型学习的性能。因此，D2Former充分利用了复数值网络运算、双路径特征处理、和联合目标训练的优势，在与之前的模型相比中，D2Former以最小的模型参数量（0.87M）在VoiceBank+Demand基准测试中取得了最好的语音增强综合效果。  


### 2022-7-25
paper title:DNN-Free Low-Latency Adaptive Speech Enhancement Based on Frame-Online Beamforming Powered by Block-Online FastMNMF  
标题： 基于块在线FastMNMF的帧在线波束形成的无DNN低时延自适应语音增强  
链接：https://arxiv.org/abs/2207.10934  
作者：Aditya Arie Nugraha,Kouhei Sekiguchi,Mathieu Fontaine,Yoshiaki Bando,Kazuyoshi Yoshii  
机构：Center for Advanced Intelligence Project (AIP), RIKEN, Japan, Graduate School of Informatics, Kyoto University, Japan, LTCI, T´el´ecom Paris, Institut Polytechnique de Paris, France, National Institute of Advanced Industrial Science and Technology (AIST), Japan  
备注：IWAENC 2022  

paper title:Inference skipping for more efficient real-time speech enhancement with parallel RNNs  
标题： 基于并行RNN的跳过推理实时语音增强  
链接：https://arxiv.org/abs/2207.11108  
作者：Xiaohuai Le,Tong Lei,Kai Chen,Jing Lu  
机构： Institute of Acoustics, Nanjing University  
备注：11 pages, 8 figures, accepted by IEEE/ACM TASLP  
 
### 2022-7-18
paper title：Direction-Aware Adaptive Online Neural Speech Enhancement with an Augmented Reality Headset in Real Noisy Conversational Environments  
标题：真实噪声环境下基于增强现实耳机的方向感知自适应在线神经语音增强  
链接：https://arxiv.org/abs/2207.07296  
作者：Kouhei Sekiguchi,Aditya Arie Nugraha,Yicheng Du,Yoshiaki Bando,Mathieu Fontaine,Kazuyoshi Yoshii  
备注：IEEE/RSJ IROS 2022  

### 2022-7-1
paper title：A light-weight full-band speech enhancement model  
标题：一种轻量级全频带语音增强模型  
链接：https://arxiv.org/abs/2206.14524  
作者：Qinwen Hu,Zhongshu Hou,Xiaohuai Le,Jing Lu  
机构： Key Laboratory of Modern Acoustics, Nanjing University, Nanjing, China  

paper title：ClearBuds: Wireless Binaural Earbuds for Learning-Based Speech Enhancement  
标题：ClearBuds：用于基于学习的语音增强的无线双耳耳机  
链接：https://arxiv.org/abs/2206.13611  
作者：Ishan Chatterjee,Maruchi Kim,Vivek Jayaram,Shyamnath Gollakota,Ira Kemelmacher-Shlizerman,Shwetak Patel,Steven M. Seitz  
机构：∗Co-primary student authors, Paul G. Allen School of Computer Science & Engineering, University of Washington, Seattle, WA, USA  
备注：12 pages, Published in Mobisys 2022  

### 2022-5-5
paper title：Improving Dual-Microphone Speech Enhancement by Learning Cross-Channel Features with Multi-Head Attention  
标题：基于多头注意的跨通道特征学习改进双麦克风语音增强  
作者：Xinmeng Xu,Rongzhi Gu,Yuexian Zou  
机构*：ADSPLAB, School of ECE, Peking University, Shenzhen, China, Peng Cheng Laboratory, Shenzhen, China  
备注：Accepted by ICASSP 2022  

### 2022-3-24
paper title：Joint Noise Reduction and Listening Enhancement for Full-End Speech Enhancement  
标题：基于联合降噪和听力增强的全端语音增强  
作者：Haoyu Li,Yun Liu,Junichi Yamagishi  
机构*：National Institute of Informatics, Japan, The Graduate University for Advanced Studies (SOKENDAI), Japan  
备注：Submitted to Interspeech 2022  
paper links:https://arxiv.org/pdf/2203.11500v1.pdf  

### 2022-3-17
paper title：FB-MSTCN: A Full-Band Single-Channel Speech Enhancement Method Based on Multi-Scale Temporal Convolutional Network  
标题：FB-MSTCN：一种基于多尺度时域卷积网络的全频带单通道语音增强方法  
作者：Zehua Zhang,Lu Zhang,Xuyi Zhuang,Yukun Qian,Heng Li,Mingjiang Wang  
机构*：Harbin Institute of Technology, Shenzhen, China  
备注：Accepted by ICASSP 2022, Deep Noise Suppression Challenge  
链接：https://arxiv.org/pdf/2203.07684v1.pdf  

paper title：TaylorBeamformer: Learning All-Neural Multi-Channel Speech Enhancement from Taylor's Approximation Theory  
标题：泰勒波束形成器：从泰勒近似理论学习全神经多通道语音增强  
作者：Andong Li,Guochen Yu,Chengshi Zheng,Xiaodong Li  
机构*：Key Laboratory of Noise and Vibration Research, Institute of Acoustics, Chinese Academy of, University of Chinese Academy of Sciences, Beijing, China  
备注：5 pages, submitted to Interspeech2022  
链接：https://arxiv.org/pdf/2203.07195v1.pdf  

paper title：MDNet: Learning Monaural Speech Enhancement from Deep Prior Gradient  
标题：MDNet：从深度先验梯度学习单声道语音增强  
作者：Andong Li,Chengshi Zheng,Ziyang Zhang,Xiaodong Li  
机构*：Key Laboratory of Noise and Vibration Research, Institute of Acoustics, Chinese Academy of, University of Chinese Academy of Sciences, Beijing, China, Advanced Computing and Storage Lab, Huawei Technologies Co. Ltd., Beijing, China  
备注：5 pages, Submitted to Interspeech2022  
链接：https://arxiv.org/pdf/2203.07179v1.pdf  


### 2022-3-11
paper title：PercepNet+: A Phase and SNR Aware PercepNet for Real-Time Speech Enhancement  
标题：PercepNet+：一种相位和信噪比感知的实时语音增强PercepNet  
作者：Xiaofeng Ge,Jiangyu Han,Yanhua Long,Haixin Guan  
机构*：Shanghai Normal University, Shanghai, China, Unisound AI Technology Co., Ltd., Beijing, China  
备注：This article was submitted to Interspeech 2022  
paper link: https://arxiv.org/pdf/2203.02263v1.pdf  

paper title：Integrating Statistical Uncertainty into Neural Network-Based Speech Enhancement  
标题：统计不确定性与神经网络语音增强的融合  
作者：Huajian Fang,Tal Peer,Stefan Wermter,Timo Gerkmann  
机构*：Signal Processing (SP), Universit¨at Hamburg, Germany, Knowledge Technology (WTM), Universit¨at Hamburg, Germany  
Journal-ref：ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)  
paper link: https://arxiv.org/pdf/2203.02288v1.pdf  

### 2022-3-7
paper title：Deep Learning-Based Joint Control of Acoustic Echo Cancellation, Beamforming and Postfiltering  
标题：基于深度学习的声学回波抵消、波束形成和后滤波联合控制  
作者：Thomas Haubner,Walter Kellermann  
机构*：Multimedia Communications and Signal Processing, Friedrich-Alexander-University Erlangen-N¨urnberg  
备注：Submitted to European Signal Processing Conference (EUSIPCO) 2022, Belgrade, Serbia  
paper link: https://arxiv.org/pdf/2203.01793v1.pdf  

paper title：DMF-Net: A decoupling-style multi-band fusion model for real-time full-band speech enhancement  
标题：DMF-Net：一种解耦的实时全频带语音增强多带融合模型  
作者：Guochen Yu,Yuansheng Guan,Weixin Meng,Chengshi Zheng,Hui Wang  
机构*：⋆ State Key Laboratory of Media Convergence and Communication, Communication University, of China, Beijing, China, † Institute of Acoustics, Chinese Academy of Sciences, Beijing, China  
备注：submitted to Eusipco2022; 5 pages  
paper link: https://arxiv.org/pdf/2203.00472v1.pdf  


### 2022-2-16
paper title：Low-latency Monaural Speech Enhancement with Deep Filter-bank Equalizer  
标题：基于深度过滤银行均衡器的低延迟单声道语音增强  
作者：Chengshi Zheng,Wenzhe Liu,Andong Li,Yuxuan Ke,Xiaodong Li  
备注：35 pages, 8 figures  
链接：https://arxiv.org/pdf/2202.06764v1.pdf  


### 2022-2-9
paper title：A two-step backward compatible fullband speech enhancement system  
标题：一种两步后向兼容的全带语音增强系统  
作者：Xu Zhang,Lianwu Chen,Xiguang Zheng,Xinlei Ren,Chen Zhang,Liang Guo,Bing Yu  
机构*：Kuaishou Technology, Beijing, China  
link:https://arxiv.org/pdf/2201.10809v1.pdf  

### 2022-1-25
推荐值：AAAAA
paper title：A Dual-branch attention-in-attention transformer for single-channel SE  
code:https://github.com/YoungJay0612/DB-AIAT  
机构： 中科院 字节跳动  

### 2021-12-15
paper title:  U-shaped Transformer with Frequency-Band Aware Attention for Speech Enhancement  
标题：用于语音增强的频带感知U型Transformer  
链接：https://arxiv.org/abs/2112.06052  
作者：Yi Li,Yang Sun,Syed Mohsen Naqvi  
机构： Newcastle University , University of Oxford  

### 2021-10-14
paper title: Dual-branch Attention-In-Attention Transformer for single-channel speech enhancement  
paper title: 用于单通道语音增强的双分支注意-注意转换器  
链接：https://arxiv.org/abs/2110.06467  
作者：Guochen Yu,Andong Li,Yutian Wang,Yinuo Guo,Hui Wang,Chengshi Zheng  
机构：⋆ State Key Laboratory of Media Convergence and Communication, Communication University, of China, Beijing, China, † Institute of Acoustics, Chinese Academy of Sciences, Beijing, China, ∗ Bytedance, Beijing, China  
备注：Submitted to ICASSP 2022  

### 2021-10-13
paper title: MetricGAN-U: Unsupervised speech enhancement/ dereverberation based only on noisy/ reverberated speech  
MetricGAN-U：仅基于噪声/混响语音的无监督语音增强/去混响  
链接：https://arxiv.org/abs/2110.05866  
作者：Szu-Wei Fu,Cheng Yu,Kuo-Hsuan Hung,Mirco Ravanelli,Yu Tsao  
机构： Research Center for Information Technology Innovation, Academia Sinica, Taipei, Taiwan, Mila-Quebec AI Institute, Montreal, Canada  

paper title: Foster Strengths and Circumvent Weaknesses: a Speech Enhancement Framework with Two-branch Collaborative Learning  
扬长避短：两分支协作学习的语音增强框架  
链接：https://arxiv.org/abs/2110.05713  
作者：Wenxin Tai,Jiajia Li,Yixiang Wang,Tian Lan,Qiao Liu  
机构：University of Electronic Science and Technology of China  

DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering  
DeepFilterNet：一种基于深度滤波的低复杂度全带语音增强框架  
链接：https://arxiv.org/abs/2110.05588  
作者：Hendrik Schröter,Alberto N. Escalante-B.,Tobias Rosenkranz,Andreas Maier  
机构：Friedrich-Alexander-Universit¨at Erlangen-N¨urnberg, Pattern Recognition Lab, WS Audiology, Research and Development, Erlangen, Germany  

### 2021-10-11

paper title: Leveraging Low-Distortion Target Estimates for Improved Speech Enhancement  
标题：利用低失真目标估计来改进语音增强  
链接：https://arxiv.org/abs/2110.00570  
作者：Zhong-Qiu Wang,Gordon Wichern,Jonathan Le Roux  
机构： He is now withthe Language Technologies Institute, Carnegie Mellon University  
备注：in submission  

paper title: End-to-End Complex-Valued Multidilated Convolutional Neural Network for Joint Acoustic Echo Cancellation and Noise Suppression  
标题：端到端复值倍增卷积神经网络用于声学回波抵消和噪声抑制  
链接：https://arxiv.org/abs/2110.00745  
作者：Karn N. Watcharasupat,Thi Ngoc Tho Nguyen,Woon-Seng Gan,Shengkui Zhao,Bin Ma  
机构：⋆School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore, †Speech Lab, Alibaba Group, Singapore  
备注：Submitted to the 2022 International Conference on Acoustics, Speech, & Signal Processing (ICASSP)  

paper title: Lightweight Speech Enhancement in Unseen Noisy and Reverberant Conditions using KISS-GEV Beamforming  
标题：KISS-GEV波束形成在不可见噪声和混响条件下的轻量级语音增强  
链接：https://arxiv.org/abs/2110.03103  
作者：Thomas Bernard,Cem Subakan,François Grondin  
机构：Universit´e de Sherbrooke, Sherbrooke (Qu´ebec), Canada  

paper title: On audio enhancement via online non-negative matrix factorization  
标题：基于在线非负矩阵分解的音频增强研究  
链接：https://arxiv.org/abs/2110.03114  
作者：Andrew Sack,Wenzhao Jiang,Michael Perlmutter,Palina Salanevich,Deanna Needell  
code: https://github.com/Jerry-jwz/Audio-Enhancement-via-ONMF  

paper title: Embedding and Beamforming: All-neural Causal Beamformer for Multichannel Speech Enhancement  
标题：嵌入和波束形成：用于多通道语音增强的全神经因果波束形成器  
链接：https://arxiv.org/abs/2109.00265  
作者：Andong Li,Wenzhe Liu,Chengshi Zheng,Xiaodong Li  
机构：⋆ Key Laboratory of Noise and Vibration Research, Institute of Acoustics, Chinese Academy, † University of Chinese Academy of Sciences, Beijing, China  
备注：Submitted to ICASSP 2022, first version  

### 2021-7-27
paper title: Inplace Gated Convolutional Recurrent Neural Network For Dual-channel Speech Enhancement  
标题：用于双通道语音增强的就地门控卷积递归神经网络  
作者：Jinjiang Liu,Xueliang Zhang  
机构：College of Computer Science, Inner Mongolia University, China  
备注：Accepted by INTERSPEECH2021  
链接：https://arxiv.org/abs/2107.11968  

### 2021-7-13
paper title: DPCRN: Dual-Path Convolution Recurrent Network for Single Channel Speech Enhancement  
标题：DPCRN：用于单通道语音增强的双径卷积递归网络  
作者：Xiaohuai Le,Hongsheng Chen,Kai Chen,Jing Lu  
机构：Key Laboratory of Modern Acoustics, Nanjing University, Nanjing , China, NJU-Horizon Intelligent Audio Lab, Horizon Robotics, Beijing , China, Nanjing Institute of Advanced Artificial Intelligence, Nanjing , China  
备注：5 pages, 1 figure, accepted by Interspeech 2021  
链接：https://arxiv.org/abs/2107.05429  


### 2021-6-17
paper title: A Flow-Based Neural Network for Time Domain Speech Enhancement  
标题：一种基于流的时域语音增强神经网络  
作者：Martin Strauss,Bernd Edler  
机构：International Audio Laboratories Erlangen, Am Wolfsmantel , Erlangen, Germany  
备注：Accepted to ICASSP 2021  
链接：https://arxiv.org/abs/2106.09008  

paper title: DCCRN+: Channel-wise Subband DCCRN with SNR Estimation for Speech Enhancement  
标题：DCCRN+：用于语音增强的信噪比估计信道子带DCCRN  
作者：Shubo Lv,Yanxin Hu,Shimin Zhang,Lei Xie  
机构：Northwestern Polytechnical University, Xi'an, China  
链接：https://arxiv.org/abs/2106.08672  

###2021-6-15
paper title: F-T-LSTM based Complex Network for Joint Acoustic Echo Cancellation and Speech Enhancement  
标题：基于F-T-LSTM的声学回波抵消和语音增强联合复杂网络  
作者：Shimin Zhang,Yuxiang Kong,Shubo Lv,Yanxin Hu,Lei Xie  
机构：Northwestern Polytechnical University, Xi’an, China  
备注：submitted to Interspeech 2021  
链接：https://arxiv.org/abs/2106.07577  

### 2021-6-11
paper title: MetricGAN: Generative Adversarial Networks based Black-box Metric Scores Optimization for Speech Enhancement  
paper link: https://arxiv.org/pdf/1905.04874v1.pdf  
paper code: https://github.com/JasonSWFu/MetricGAN
MetricGAN+: https://github.com/speechbrain/speechbrain/tree/develop/recipes/Voicebank/enhance/MetricGAN



### 2021-5-28 - 2021-6-5
paper title: Noise Classification Aided Attention-Based Neural Network for Monaural Speech Enhancement  
标题：噪声分类辅助的基于注意力的神经网络单声道语音增强  
作者：Lu Ma,Song Yang,Yaguang Gong,Zhongqin Wu  
机构*：TAL Education Group, Beijing, China  
paper link: https://arxiv.org/pdf/2105.14719v1.pdf  

paper title:  Self-attending RNN for Speech Enhancement to Improve Cross-corpus Generalization  
标题：提高跨语料库泛化能力的自参加RNN语音增强算法  
作者：Ashutosh Pandey,DeLiang Wang  
机构*： Wang is with the Department of Computer Science and Engineeringand the Center for Cognitive and Brain Sciences  
备注：submitted to IEEEACM Transactions on Audio, Speech and Language Processing  
paper link: https://arxiv.org/pdf/2105.12831v1.pdf  

### 2021-5-27  
Wav2Vec预训练模型调用：  
1：参考PFP loss以及Wav2Vec调用：https://github.com/YoungJay0612/PhoneFortifiedPerceptualLoss_SE  
2：Wav2Vec2.0预训练模型调用: https://github.com/YoungJay0612/speech-representations  
3: FB Wav20Vec模型: https://github.com/pytorch/fairseq/tree/master/examples/wav2vec  


### 2021-5-27
paper title: Training Speech Enhancement Systems with Noisy Speech Datasets  
标题：利用含噪语音数据集训练语音增强系统  
作者：Koichi Saito,Stefan Uhlich,Giorgio Fabbro,Yuki Mitsufuji  
机构： The University of Tokyo, Japan, Sony Corporation, R&D Center, GermanyJapan  
备注：5 pages, 3 figures, submitted to WASPAA2021  
paper link: https://arxiv.org/pdf/2105.12315.pdf   

### 2021-5-26
paper title: RNNoise-Ex: Hybrid Speech Enhancement System based on RNN and Spectral Features  
标题：RNNoise-Ex：基于RNN和频谱特征的混合语音增强系统  
作者：Constantine C. Doumanidis,Christina Anagnostou,Evangelia-Sofia Arvaniti,Anthi Papadopoulou  
机构：School of Electrical and Computer Engineering, Aristotle University of Thessaloniki, Thessaloniki, Greece, Note: Author name order was decided arbitrarily.  
备注：6 pages, 5 figures, presented at ECESCON 12, for code see this https URL  
code links: https://github.com/CedArctic/rnnoise-ex  


### 2021-5-18
paper title: Dual-Stage Low-Complexity Reconfigurable Speech Enhancement  
标题：两级低复杂度可重构语音增强算法  
作者：Jun Yang,Nico Brailovsky  
机构：Facebook Reality Labs, Hacker Way, Menlo Park, CA , USA  
备注：5 pages  
paper link: https://arxiv.org/ftp/arxiv/papers/2105/2105.07632.pdf  

### 2021-5-8
paper title: Speech Enhancement using Separable Polling Attention and Global Layer Normalization followed with PReLU  
标题：基于PReLU的可分离轮询注意力和全局层归一化语音增强  
paper link: https://arxiv.org/pdf/2105.02509.pdf  
Authors and Publishers:   Tecent & 2Beijing Forestry University   

paper title: DBNet: A Dual-branch Network Architecture Processing on Spectrum and Waveform for Single-channel Speech Enhancement  
标题：DBNet：一种单通道语音增强频谱和波形处理的双分支网络结构  
paper link: https://arxiv.org/pdf/2105.02436.pdf  
Authors and Publishers:   Inner Mongolia University   


### 2021-4-29
paper title: DPT-FSNet:Dual-path Transformer Based Full-band and Sub-band Fusion Network for Speech Enhancement  
标题：DPT-FSNET：基于双路Transformer的全带和子带语音增强融合网络  
paper link: https://arxiv.org/pdf/2104.13002.pdf  
Authors and Publishers:  中科院  

### 2021-4-23
paper title: Nonlinear Spatial Filtering in Multichannel Speech Enhancement 非线性空间滤波在多通道语音增强中的应用  
paper link: https://arxiv.org/pdf/2104.11033.pdf  


### 2021-4-14
paper title: Complex Spectral Mapping With Attention Based Convolution Recrrent Neural Network for Speech Enhancement 基于注意力的卷积神经网络复谱映射语音增强  
Authors and Publishers:   云从科技 submitted to INTERSPEECH 2021  
paper link: https://arxiv.org/pdf/2104.05267.pdf     

### 2021-4-12
paper title: Joint Online Multichannel Acoustic Echo Cancellation, Speech Dereverberation and Source Separation 联合在线多通道声学回波对消、语音去混响和源分离  
Authors and Publishers:   Alibaba QiangFu submitted to INTERSPEECH 2021  
paper link: https://arxiv.org/pdf/2104.04325.pdf   


### 2021-4-9
paper title: Speech Denoising without Clean Training Data: a Noise2Noise Approach 无干净训练数据的语音去噪：Noise2Noise方法  
Authors and Publishers: Department of Computer Science and Engineering, PES University 法国 Sumbitted to INTERSPEECH 2021  
paper link: https://arxiv.org/pdf/2104.03838.pdf  
code links: https://github.com/madhavmk/Noise2Noise-audio_denoising_without_clean_training_data  
description： 克服了基于深度学习的语音去噪方法对干净语音数据依赖性大的障碍，证明了仅使用有噪声的语音样本就可以训练深度语音去噪网络。  此外，在涉及复杂噪声分布和低信噪比的情况下，仅使用有噪声的音频目标的训练机制比使用干净的音频目标的传统训练机制具有更好的去噪性能。  

paper title: Phoneme-based Distribution Regularization for Speech Enhancement 基于音素的分布正则化在语音增强中的应用  
Authors and Publishers:   USTC&MRA  ICASSP 2021  
paper link: https://arxiv.org/pdf/2104.03759.pdf  

paper title: MetricGAN+: An Improved Version of MetricGAN for Speech Enhancement MetricGAN+：语音增强的MetricGAN改进版本  
Authors and Publishers:   Yu Tsao https://github.com/speechbrain/speechbrain  
paper link: https://arxiv.org/ftp/arxiv/papers/2104/2104.03538.pdf    

### 202104-6
paper title: Real-time Streaming Wave-U-Net with Temporal Convolutions for Multichannel Speech Enhancement 用于多通道语音增强的带时间卷积的实时流式Wave-U网    
Authors and Publishers:   HuaWei  Submitted to InterSpeech2021  
paper link: https://arxiv.org/pdf/2104.01923.pdf  


### 2021-4-2
paper title: Y^2-Net FCRN for Acoustic Echo and Noise Suppression 用于声学回波和噪声抑制的净FcRN  
Authors and Publishers:   Institute for Communications Technology, Technische Universitat Braunschweig, Schleinitzstr. , Braunschweig, Germany  
paper link: https://arxiv.org/pdf/2103.17189.pdf  

### 2021-3-31
paper title:Time-domain Speech Enhancement with Generative Adversarial Learning 基于生成对抗学习的时域语音增强  
paper link: https://arxiv.org/pdf/2103.16149.pdf  
Authors and Publishers: Harbin Engineering University, ByteDance  


### 2021-3-30  
paper title:Channel-Attention Dense U-Net for Multichannel Speech Enhancement  
paper link: https://arxiv.org/pdf/2102.04198v1.pdf  
paper_with_code:https://paperswithcode.com/sota/speech-enhancement-on-chime-3  
Authors and Publishers: Harvard University, Amazon & MIT  
code link:  

### 2021-03-20  
paper title: Decoupling Magnitude and Phase Optimization with a Two-Stage Deep Network  
paper link: https://arxiv.org/pdf/2102.04198v1.pdf  
Authors and Publishers: 中科院 Li Andong DNS2021 No.1  
code link:  

paper title: A Recursive Network with Dynamic Attention for Monaural Speech Enhancement  
paper link: https://arxiv.org/abs/2003.12973  
Authors and Publishers: 中科院  Li Andong  InterSpeech2020  
code link: https://github.com/Andong-Li-speech/DARCN  

paper title: Interactive Speech and Noise Modeling for Speech Enhancement  
paper links:　https://arxiv.org/pdf/2012.09408.pdf   
Authors and Publishers: Microsoft AAAI 2021  
remark：首次提出交互式语音和噪声分离模型，也可以应用到语音分离等其他多信号处理的任务    
code link:  

paper title: CAUNet: Context-Aware UNet for Speech Enhancement in Time Domain  
paper links:　https://github.com/key2miao/CAUNet  
Authors and Publishers: Concordia University, Montreal, Canada  ISCAS 2021  
code link:  

paper title: End to End Waveform Utterance Enhancement for Direct Evaluation Metrics Optimization by Fully Convolutional Neural Networks  
Authors and Publishers: 台湾中央研究院陶宇 2018 TASLP  
remark：SpeechBrain中的wav2wav语音增强所使用的方法  
code link:  

paper title:　Monaural Speech Enhancement with Complex Convolutional Block Attention Module and Joint Time Frequency Losses  
paper links:　https://arxiv.org/pdf/2102.01993v1.pdf  
Authors and Publishers: 　Alibaba group  
record reason:　在PapersWithCode中的DNS数据集任务中，表现最好，优于DCCRN和DCUnet  
code link:　https://paperswithcode.com/sota/speech-enhancement-on-interspeech-2020-deep  
remark：　目前没有开源代码

### 2021-03-19  
paper title:　TSTNN: Two-stage Transformer based Neural Network for Speech Enhancement in the Time Domain  
paper links:　https://arxiv.org/ftp/arxiv/papers/2103/2103.09963.pdf  
Authors and Publishers: 　Concordia University, Montreal, Canada   ICASSP 2021
record reason:　Transformer结构应用于语音增强  
code link:  https://github.com/key2miao/TSTNN  

paper title:　Dense CNN with Self-Attention for Time-Domain Speech Enhancement  
paper links:　https://arxiv.org/pdf/2009.01941.pdf  
Authors and Publishers: 　WangDeLiang
record reason:　能解决demucs时域增强的杂音问题？？？  
code link:  
remark：　目前没有开源代码

### Earlier  
#### Github  
facebook demucs：https://github.com/facebookresearch/denoiser  
paper title:
remark：facebook开源的时域语音增强  
  
PFP loss: https://github.com/YoungJay0612/PhoneFortifiedPerceptualLoss_SE  
paper title: Improving Perceptual Quality by Phone-Fortified Perceptual Loss for Speech" Enhancement"  
remark:台湾中央研究院陶宇实验室论文，结合音素  

PAHSEN： https://github.com/YoungJay0612/Microsoft-Phasen-SE  
pape rtitle: PHASEN: A Phase-and-Harmonics-Aware Speech Enhancement Network  
Authors and Publishers: 　Microsoft  
remark: Unofficial PyTorch implementation  

FullSubnet: https://github.com/haoxiangsnr/FullSubNet  
pape rtitle: FullSubNet: A Full-Band and Sub-Band Fusion Model for Real-Time Single-Channel Speech Enhancement  
Authors and Publishers: 　HaoXiang ICASSP2021  
remark: Official PyTorch implementation  

WaveCRN: https://github.com/YoungJay0612/WaveCRN
pape rtitle: An Efficient Convolutional Recurrent Neural Network for End-to-end Speech Enhancement
remark: Unofficial PyTorch implementation  

DCUnet:https://github.com/YoungJay0612/DeepComplexUNetPyTorch  
paper title:Phase-Aware Speech Enhancement with Deep Complex U-Net  
remark: Unofficial PyTorch implementation  

基于深度特征映射的语音增强方法：
https://github.com/linan2/TensorFlow-speech-enhancement-Chinese

ICASSP2021:
Seoul National Unicersity: Real-time Denoising and Dereverberation with Tiny Recurrent U-Net    
Facebook：【基于图神经网络的多通道语音增强】Multi-Channel Speech Enhancement Using Graph Neural Networks

## Speech Dereverberation
### 2022-11-27
title: SkipConvGAN: Monaural Speech Dereverberation using Generative Adversarial Networks via Complex Time-Frequency Masking  
标题：SkipConvGAN：基于复杂时频掩蔽的生成对抗性网络单声道语音去混响  
链接：https://arxiv.org/abs/2211.12623  
作者：Vinay Kothapally,J. H. L. Hansen  
机构：University of Texas at Dallas  
备注：Published in: IEEE/ACM Transactions on Audio, Speech, and Language Processing ( Volume: 30)  

title: Complex-Valued Time-Frequency Self-Attention for Speech Dereverberation  
标题：用于语音去混响的复值时频自注意  
链接：https://arxiv.org/abs/2211.12632  
作者：Vinay Kothapally,John H. L. Hansen  
机构：Center for Robust Speech Systems (CRSS), The University of Texas at Dallas, TX, USA  
备注：Interspeech 2022: ISCA Best Student Paper Award Finalist  

### 2022-6-17
paper title：To Dereverb Or Not to Dereverb? Perceptual Studies On Real-Time Dereverberation Targets  
标题：去混响还是不去混响？实时去混响目标的感知研究  
链接：https://arxiv.org/abs/2206.07917  
作者：Jean-Marc Valin,Ritwik Giri,Shrikant Venkataramani,Umut Isik,Arvindh Krishnaswamy  
机构：Amazon Web Services, Palo Alto, CA, USA  
备注：5 pages  

### 2022-5-20
paper title：Utterance Weighted Multi-Dilation Temporal Convolutional Networks for Monaural Speech Dereverberation  
标题：用于单声道语音去混响的发音加权多伸缩时间卷积网络  
链接：https://arxiv.org/abs/2205.08455  
作者：William Ravenscroft,Stefan Goetze,Thomas Hain  
机构：Department of Computer Science, The University of Sheffield , Sheffield, United Kingdom  
备注：Submitted to IWAENC 2022  
github links: https://github.com/jwr1995/DF-TCN  

### 2022-4-20
paper title：Single-Channel Speech Dereverberation using Subband Network with A Reverberation Time Shortening Target  
标题：以缩短混响时间为目标的子带网络单通道语音去混响  
作者：Rui Zhou,Wenye Zhu,Xiaofei Li  
机构*：Westlake University & Westlake Institute for Advanced Study, Hangzhou, China, Zhejiang University, Hangzhou, China  
备注：Submitted to INTERSPEECH 2022  
paper links: https://arxiv.org/pdf/2204.08765v1.pdf  

### 2021-10-19
paper title：Controllable Multichannel Speech Dereverberation based on Deep Neural Networks  
标题：基于深度神经网络的可控多通道语音去混响  
链接：https://arxiv.org/abs/2110.08439  
作者：Ziteng Wang,Yueyue Na,Biao Tian,Qiang Fu  
机构：Alibaba Group, China  
备注：submitted to ICASSP2022  

### 2021-10-12
paper title:Late reverberation suppression using U-nets  
标题：基于U网的延迟混响抑制  
链接：https://arxiv.org/abs/2110.02144  
作者：Diego León,Felipe Tobar  
机构：⋆Department of Electrical Engineering, Universidad de Chile, †Initiative for Data & Artificial Intelligence, Universidad de Chile, ‡Center for Mathematical Modeling, Universidad de Chile  


REVERB challenge: http://reverb2014.dereverberation.com/  

### 2021-7-7
paper title:Skip Convolutional Neural Network for Speech Dereverberation using Optimally Smoothed Spectral Mapping  
paper links: https://arxiv.org/pdf/2007.09131.pdf  
paper website: https://vkothapally.github.io/SkipConv/  
github repository: https://github.com/zehuachenImperial/SkipConvNet  

pape rtitle: TeCANet Temporal-Contextual Attention Network for Environment-Aware Speech Dereverberation  
TeCANet：支持环境感知语音去混响的时间-上下文注意网络  
Authors and Publishers: Peking University, Shenzhen Tencent AI Lab  
remark: ：Submitted to Interspeech 2021  
paper links:https://arxiv.org/pdf/2103.16849.pdf  

NAra_WPE: https://github.com/fgnt/nara_wpe  
DNN_WPE: https://github.com/nttcslab-sp/dnn_wpe  
