# 1. Base Image: NVIDIA PyTorch optimized for Blackwell
FROM nvcr.io/nvidia/pytorch:25.01-py3

# 2. Prevent Interactive Prompts
ENV DEBIAN_FRONTEND=noninteractive

# 3. Fix Optree & Architecture Recognition
ENV TORCH_CUDA_ARCH_LIST="10.0;12.0"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 4. System Dependencies (Noble-compatible replacement)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1 \
    libglx0 \
    && rm -rf /var/lib/apt/lists/*

# 5. Full Library Stack (No more !pip install)
# Ensure compatibility with CUDA 12.8-compatible versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir \
    'optree>=0.13.0' \
    'transformers>=4.48.0' \
    'sentence-transformers>=3.3.0' \
    'accelerate>=1.2.0' \
    datasets \
    plotly \
    streamlit \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    shap \
    nbformat \
    ipywidgets


# 6. Pre-load SBERT 
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

WORKDIR /workspace
EXPOSE 8888 8501

# Default Command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]