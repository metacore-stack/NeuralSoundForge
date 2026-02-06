# Neural Audio Synthesis & Semantic Control Framework

A sophisticated deep learning platform for generative audio synthesis with semantic control capabilities, built using state-of-the-art StyleGAN2 architectures and perceptual guidance techniques.

## Project Overview

This advanced AI-powered system demonstrates expertise in **generative adversarial networks (GANs)**, **audio signal processing**, and **semantic feature manipulation** for creative audio design. The framework enables real-time generation and manipulation of audio textures through intuitive semantic controls, showcasing deep understanding of neural audio synthesis, latent space exploration, and perceptual audio modeling.

## Technical Architecture

### Core Technologies

- **Deep Learning Framework**: PyTorch-based StyleGAN2 implementation for audio generation
- **Audio Processing**: Librosa for spectral analysis, PGHI (Phase Gradient Heap Integration) for high-quality audio reconstruction
- **Perceptual Metrics**: LPIPS (Learned Perceptual Image Patch Similarity) for perceptual loss computation
- **Web Interface**: Streamlit-based multi-page application with real-time audio generation
- **Containerization**: Docker deployment with CUDA support for GPU acceleration

### Key Technical Features

#### 1. **Semantic-Guided Audio Generation**
   - **Perceptual Guided Control**: Advanced algorithm that enables semantic manipulation of audio through learned prototypes
   - **Multi-dimensional Control**: Independent sliders for brightness, rate, and impact type parameters
   - **Prototype-based Guidance**: Semantic prototypes (e.g., "Hits", "Scratches", "DogBark", "Rain") guide generation toward desired perceptual qualities
   - **Real-time Synthesis**: GPU-accelerated generation with sub-second latency for interactive exploration

#### 2. **StyleGAN2 Audio Synthesis**
   - **Pre-trained Models**: Specialized StyleGAN2 models trained on:
     - **Hits & Scratches**: Impact sounds and surface interactions (2-second audio clips)
     - **Environmental Sounds**: Natural and urban soundscapes (4-second audio clips)
   - **Encoder Network**: Inversion network for mapping audio to latent space
   - **Spectral Representation**: STFT-based spectrogram generation with configurable hop sizes (128/256 samples)

#### 3. **SEFA (Semantic Factorization)**
   - **Latent Space Decomposition**: Automatic discovery of semantically meaningful directions in StyleGAN's latent space
   - **Multi-dimensional Exploration**: Four independent semantic dimensions for fine-grained control
   - **Example-based Navigation**: Pre-computed example points in latent space for intuitive starting positions

#### 4. **Query-Based GAN Interface**
   - **Soft Prior Integration**: Semantic priors that guide generation toward specific sound categories
   - **Interactive Query System**: Real-time audio generation based on semantic queries
   - **Reactive Interface**: React-based frontend for enhanced user experience and performance

#### 5. **Advanced Audio Processing Pipeline**
   - **PGHI Reconstruction**: Phase Gradient Heap Integration for high-quality time-frequency reconstruction
   - **Loudness Normalization**: Pyloudnorm integration for consistent audio levels
   - **Spectral Analysis**: Real-time spectrogram visualization with librosa
   - **16kHz Sampling Rate**: Optimized for efficient processing while maintaining perceptual quality

## Interface Screenshots

### Main Interface - Semantic Control Panel

<p float="center">
  <img src="screenshots/interface-1.png" width="49%" />
  <img src="screenshots/interface-2.png" width="49%" /> 
</p>

**Left Image (Interface-1)**: The primary semantic control interface showcasing the perceptual guided control system. This view demonstrates:
- **Model Selection Sidebar**: Dropdown for choosing between specialized audio models (Hits & Scratches, Environmental Sounds)
- **Semantic Parameter Sliders**: Three independent control dimensions (Brightness, Rate, Impact Type) with range [-5.0, 5.0] for fine-grained manipulation
- **Real-time Spectrogram Visualization**: Live frequency-domain representation showing the spectral characteristics of generated audio
- **Audio Playback Controls**: Integrated audio player for immediate listening of synthesized sounds
- **Prototype Selection**: Dropdown menu for selecting semantic prototypes that guide the generation process

**Right Image (Interface-2)**: The advanced exploration interface featuring the SEFA (Semantic Factorization) algorithm. This view illustrates:
- **Multi-dimensional Latent Space Navigation**: Four independent sliders (Dimension 1-4) for exploring different semantic directions discovered through factorization
- **Example-based Starting Points**: Pre-computed example selections that provide meaningful starting positions in the latent space
- **Model-specific Controls**: Context-aware interface that adapts based on the selected audio model
- **Interactive Audio Generation**: Real-time synthesis with visual feedback through spectrogram updates
- **Session Management**: UUID-based session tracking for analytics and user experience optimization

Both interfaces demonstrate the sophisticated real-time audio generation capabilities, with seamless integration between deep learning models and user interaction. The visualizations provide immediate feedback on how semantic parameter adjustments affect the spectral characteristics of the generated audio.

## Technical Implementation Details

### Model Architecture
- **Generator Network**: StyleGAN2 architecture adapted for audio spectrogram generation
- **Encoder Network**: Inversion network for audio-to-latent mapping
- **Latent Space**: 512-dimensional W+ space for StyleGAN2, enabling semantic manipulation
- **Direction Vectors**: Pre-computed semantic directions for brightness, rate, and impact type

### Audio Processing Pipeline
1. **Input**: Latent code or semantic parameters
2. **Generation**: StyleGAN2 synthesis to spectrogram
3. **Post-processing**: Renormalization and loudness adjustment
4. **Reconstruction**: PGHI-based time-frequency reconstruction
5. **Output**: 16kHz audio waveform

### Performance Optimizations
- **GPU Acceleration**: CUDA-enabled PyTorch operations
- **Model Caching**: Streamlit caching for efficient model loading
- **Batch Processing**: Optimized tensor operations for real-time generation
- **Memory Management**: Efficient handling of large pre-trained models (~11GB Docker image)

## Dependencies

### Core Libraries
- `torch` - PyTorch deep learning framework
- `librosa==0.10.0` - Audio analysis and processing
- `numpy==1.23.5` - Numerical computations
- `streamlit` - Web application framework
- `scipy` - Scientific computing utilities
- `matplotlib` - Visualization

### Specialized Audio Libraries
- `lpips==0.1.4` - Perceptual similarity metrics
- `pyloudnorm==0.1.1` - Loudness normalization
- `soundfile` - Audio I/O operations

### Additional Tools
- `typing-extensions==4.6.3` - Type hinting support
- `pandas` - Data manipulation
- `altair` - Statistical visualizations

## Deployment

### Docker Deployment

The application is containerized for easy deployment:

**Prerequisites:**
- Docker installed
- CUDA 11.0 or higher (for GPU acceleration)
- Approximately 11 GB disk space for Docker image

**Running the Application:**

```bash
docker build -t neural-audio-synthesis .
docker run -p 8100:8100 neural-audio-synthesis
```

**Available Endpoints:**
- Semantic Guided Control: `http://localhost:8100/sound_design/?app=our-algo`
- Query GAN (Classic): `http://localhost:8100/sound_design/?app=algo-1`
- SEFA Exploration: `http://localhost:8100/sound_design/?app=algo-2`
- Query GAN (React): `http://localhost:8100/sound_design/?app=algo1`
- SEFA (React): `http://localhost:8100/sound_design/?app=algo2`

### Local Development

For local development without Docker:

```bash
cd interface
streamlit run app.py --server.port=<port number>
```

## Skills Demonstrated

This project showcases advanced expertise in:

- **Deep Learning & Neural Networks**: Implementation and optimization of StyleGAN2 for audio generation
- **Generative AI**: Advanced GAN architectures, latent space manipulation, and semantic control
- **Audio Signal Processing**: Spectral analysis, phase reconstruction, and perceptual audio modeling
- **Machine Learning Engineering**: Model deployment, optimization, and production-ready implementations
- **Full-Stack Development**: Streamlit web applications, React frontends, and Docker containerization
- **Research & Innovation**: Semantic factorization techniques, perceptual guidance, and novel audio synthesis methods
- **Software Architecture**: Modular design, multi-page applications, and scalable system architecture

## Technical Highlights

- **State-of-the-Art Models**: Pre-trained StyleGAN2 models specialized for different audio domains
- **Real-time Performance**: GPU-accelerated generation enabling interactive exploration
- **Semantic Control**: Intuitive parameter spaces mapped to perceptual audio qualities
- **Production-Ready**: Docker containerization, error handling, and session management
- **Extensible Architecture**: Modular design allowing easy addition of new models and algorithms

---

*This project represents a comprehensive implementation of cutting-edge neural audio synthesis techniques, combining deep learning research with practical software engineering for creative audio applications.*
