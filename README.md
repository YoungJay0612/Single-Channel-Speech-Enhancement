# Speech-Processing-Papers-Code(持续更新中)
Keep track of good articles on speech processing, mainly on speech enhancement, include speech denoise, speech dereverberation and aec、agc, etc.

## Speech Denoise

### 2021-4-12



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
code link:  
remark：　目前没有开源代码

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

pape rtitle: TeCANet Temporal-Contextual Attention Network for Environment-Aware Speech Dereverberation  
TeCANet：支持环境感知语音去混响的时间-上下文注意网络  
Authors and Publishers: Peking University, Shenzhen Tencent AI Lab  
remark: ：Submitted to Interspeech 2021  
paper links:https://arxiv.org/pdf/2103.16849.pdf  

NAra_WPE: https://github.com/fgnt/nara_wpe  
DNN_WPE: https://github.com/nttcslab-sp/dnn_wpe  

## Speech Separation

2021-3-26
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
DTLN: https://github.com/YoungJay0612/DTLN-aec-TensorFlow  
RNN_AEC: https://github.com/shichaog/RNNAec 葛世超

### Speech Quality Assessment

paper title: MetricNet: Towards Improved Modeling For Non-Intrusive Speech Quality Assessment  MetricNet：面向非侵入式语音质量评估的改进建模  
paper links: https://arxiv.org/pdf/2104.01227.pdf  
Authors and Publishers: Tencent AI Lab  


