# Speech-Processing-Papers-Code(持续更新中)
Keep track of good articles on speech processing, mainly on speech enhancement, include speech denoise, speech dereverberation and aec、agc, etc.

## Speech Denoise

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




