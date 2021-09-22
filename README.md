# Speech-Processing-Papers-Code(持续更新中)
Keep track of good articles on speech processing, mainly on speech enhancement, include speech denoise, speech dereverberation and aec、agc, etc.

## Speech Deep Learning

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

## Speech Separation

### 2021-4-29
paper title: Complex Neural Spatial Filter: Enhancing Multi-channel Target Speech Separation in Complex Domain  
标题：复神经空间过滤：增强复数域多通道目标语音分离  
links: https://arxiv.org/pdf/2104.12359.pdf  

### 2021-4-23
paper title: Many-Speakers Single Channel Speech Separation with Optimal Permutation Training 基于最优排列训练的多说话人单通道语音分离  
Authors and Publishers: FaceBook AI Research  
links: https://arxiv.org/pdf/2104.08955.pdf  

paper title : MIMO Self-attentive RNN Beamformer for Multi-speaker Speech Separation 用于多说话人语音分离的MIMO自关注RNN波束形成器  
Authors and Publishers: Tecent AI Lab 中科院自动化所  
links: https://arxiv.org/pdf/2104.08450.pdf  

### 2021-3-26
paper title: Blind Speech Separation and Dereverberation using Neural Beamforming 基于神经波束形成的语音盲分离与去混响  
links: https://arxiv.org/pdf/2103.13443.pdf  
Authors and Publishers:  
code link:  

Wave-U-Net: https://github.com/f90/Wave-U-Net-Pytorch  
paper links: https://arxiv.org/abs/1806.03185  

ConvTasNet: https://github.com/YoungJay0612/conv-tasnet  
paper title:TasNet: Surpassing Ideal Time-Frequency Masking for Speech Separation  
remark: official PyTorch implementation  

Spleeter: https://github.com/deezer/spleeter  

## AEC

### 2021-9-22
paper title:Acoustic Echo Cancellation using Residual U-Nets  
标题：基于剩余U网的声学回波抵消  
链接：https://arxiv.org/abs/2109.09686  
作者：J. Silva-Rodríguez,M. F. Dolz,M. Ferrer,A. Castelló,V. Naranjo,G. Piñero  
机构： Universitat Politecnica de Valencia  
备注：6 pages, 2 figures, submitted to the 2021 IEEE International Conference on Acoustics, Speech and Signal Processing on October 2020  

### 2021-8-9
paper title:Deep Residual Echo Suppression and Noise Reduction: A Multi-Input FCRN Approach in a Hybrid Speech Enhancement System  
标题：深层残留回波抑制与降噪：一种混合语音增强系统中的多输入FcRN方法  
链接：https://arxiv.org/abs/2108.03051  
作者：Jan Franzen,Tim Fingscheidt  
机构：Institute for Communications Technology, Technische Universit¨at Braunschweig, Schleinitzstr. , Braunschweig, Germany  


### 2021-7-28
paper title:EchoFilter: End-to-End Neural Network for Acoustic Echo Cancellation  
机构：学而思  
paper links:https://arxiv.org/abs/2105.14666  


### 2021-6-29
paper title: Deep Residual Echo Suppression with A Tunable Tradeoff Between Signal Distortion and Echo Suppression  
标题：在信号失真和回波抑制之间进行可调折衷的深层残余回波抑制  
作者：Amir Ivry,Israel Cohen,Baruch Berdugo  ICASSP 2021  
机构：Technion – Israel Institute of Technology, Technion City, Haifa , Israel  
paper links: https://arxiv.org/pdf/2106.13531.pdf  

paper title: Nonlinear Acoustic Echo Cancellation with Deep Learning  
标题：基于深度学习的非线性声学回波抵消  
作者：Amir Ivry,Israel Cohen,Baruch Berdugo  
机构：Technion – Israel Institute of Technology, Technion City, Haifa , Israel  
备注：Accepted to Interspeech 2021  
链接：https://arxiv.org/abs/2106.13754  


### 2021-6-1
paper title: EchoFilter: End-to-End Neural Network for Acoustic Echo Cancellation  
标题：EchoFilter：端到端的声学回波消除神经网络  
作者：Lu Ma,Song Yang,Yaguang Gong,Xintian Wang,Zhongqin Wu  
机构*：TAL Education Group, Beijing, China  
备注：5 pages, 3 figures, 6 tabels  
paper links: https://arxiv.org/pdf/2105.14666v1.pdf  

### 2021-6-2
paper ttle: Multi-Scale Attention Neural Network for Acoustic Echo Cancellation  
标题：多尺度注意力神经网络在声回波消除中的应用  
作者：Lu Ma,Song Yang,Yaguang Gong,Zhongqin Wu  
机构*：TAL Education Group, Beijing, China  
paper link: https://arxiv.org/pdf/2106.00010v1.pdf  


DTLN: https://github.com/YoungJay0612/DTLN-aec-TensorFlow  
RNN_AEC: https://github.com/shichaog/RNNAec 葛世超

### Speech Quality Assessment

paper title: MetricNet: Towards Improved Modeling For Non-Intrusive Speech Quality Assessment  MetricNet：面向非侵入式语音质量评估的改进建模  
paper links: https://arxiv.org/pdf/2104.01227.pdf  
Authors and Publishers: Tencent AI Lab  

### 声源定位

### 2021-9-9
paper title:  A Review of Sound Source Localization with Deep Learning Methods  
标题：基于深度学习方法的声源定位研究综述  
链接：https://arxiv.org/abs/2109.03465  
作者：Pierre-Amaury Grumiaux,Srđan Kitić,Laurent Girin,Alexandre Guérin  
备注：Submitted to IEEE Transactions on Audio, Speech, and Language Processing  

### 2021-5-14
paper title:  Multi-target DoA Estimation with an Audio-visual Fusion Mechanism  
标题：基于视听融合机制的多目标波达方向估计  
作者：Xinyuan Qian,Maulik Madhavi,Zexu Pan,Jiadong Wang,Haizhou Li  
机构：Department of Electrical and Computer Engineering, National University of Singapore, Singapore  
备注：ICASSP 2021 accepted  
paper links: https://arxiv.org/pdf/2105.06107.pdf  

### 2021-5-6
paper title: Improved feature extraction for CRNN-based multiple sound source localization  
标题：基于CRNN的改进特征提取多声源定位  
paper link:https://arxiv.org/pdf/2105.01897.pdf  
Authors and Publishers: EUSIPCO 2021  

paper title: BeamLearning: an end-to-end Deep Learning approach for the angular localization of sound sources using raw multichannel acoustic pressure data  
标题：波束学习：基于多声道原始声压数据的端到端深度学习声源角度定位方法  
paper link: https://arxiv.org/pdf/2104.13347.pdf  

### MultiChannel Speech Enhancement

### 2021-8-20
标题： Multi-channel Continuous Speech Separation with Early Exit Transformer  
paper link: https://arxiv.org/pdf/2010.12180.pdf  
github: https://github.com/Sanyuan-Chen/CSS_with_EETransformer  

### 2021-8-9
标题：Complex-valued Spatial Autoencoders for Multichannel Speech Enhancement  
标题：用于多通道语音增强的复值空间自动编码器  
链接：https://arxiv.org/abs/2108.03130  
github: https://github.com/ModarHalimeh/COSPA  
作者：Mhd Modar Halimeh,Walter Kellermann  
机构： University of Erlangen-Nuremberg  

### 2021-7-26
paper title: Multi-channel Speech Enhancement with 2-D Convolutional Time-frequency Domain Features and a Pre-trained Acoustic Model  
标题：基于二维卷积时频域特征和预训练声学模型的多通道语音增强  
作者：Quandong Wang,Junnan Wu,Zhao Yan,Sichong Qian,Liyong Guo,Lichun Fan,Weiji Zhuang,Peng Gao,Yujun Wang  
机构：Wang, Xiaomi Corporation, Beijing, China  
备注：7 pages, 3 figures, submitted to APSIPA 2021  
链接：https://arxiv.org/abs/2107.11222  

### 2021-6-11
paper title: Joint Multi-Channel Dereverberation and Noise Reduction Using a Unified Convolutional Beamformer With Sparse Priors  
标题：基于稀疏先验统一卷积波束形成器的多通道联合去混响降噪  
作者：Henri Gode,Marvin Tammen,Simon Doclo  
机构：Department of Medical Physics and Acoustics and Cluster of Excellence Hearing,all, University of Oldenburg, Germany  
备注：ITG Conference on Speech Communication  
paper link: https://arxiv.org/pdf/2106.01902.pdf

### 2021-5-14
paper title: Attention-based Neural Beamforming Layers for Multi-channel Speech Recognition  
标题：基于注意力的神经波束形成层在多通道语音识别中的应用  
作者：Bhargav Pulugundla,Yang Gao,Brian King,Gokce Keskin,Harish Mallidi,Minhua Wu,Jasha Droppo,Roland Maas  
机构：Amazon Inc., USA  
paper link: https://arxiv.org/pdf/2105.05920.pdf  
