# Bengali Speech-to-Text API

High-accuracy Bengali ASR API powered by the ai4bharat/indic-conformer-600m-multilingual model using RNNT decoding.
This service provides a simple, reliable endpoint for transcribing Bengali audio files (WAV, MP3, M4A) in real time.

## 🛠️ Installation & Setup

Follow these steps strictly to set up the project on your local machine or server.

### 1. Clone the Repository
Download the project code to your machine.
```bash
git clone https://github.com/Khairul-islam99/Indic-STT.git
cd Indic-STT
```
### 2. Create Virtual Environment
It is recommended to use Conda to manage dependencies and avoid conflicts.
```bash
conda create -n bengali-asr-api python=3.11 -y
conda activate bengali-asr-api
```
## 3. Install PyTorch (Crucial Step)
You must install the PyTorch version that matches your hardware BEFORE installing other requirements. Run ONE of the following commands:

For NVIDIA GPU (CUDA 12.x) - Recommended Use this for modern GPUs (RTX 30xx, 40xx, A100, H100).
```bash
# For CUDA 12.4 - adjust based on server CUDA version
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```
### 4. Install Dependencies
Now install the remaining Python libraries.
```bash
pip install -r requirements.txt
```
## 🚀 Usage
1. Start the Server
Run the main application file. The server will initialize the model (this may take a minute) and start listening for requests.
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
2. API Documentation
Access the interactive Swagger UI to test endpoints manually:
 URL: http://localhost:8000/docs
