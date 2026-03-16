# 🌌 Galaxy Ring Classifier

Streamlit web app for classifying galactic ring structures (inner / outer / inner+outer) using a fine-tuned Zoobot ConvNeXt-nano encoder.

## Features

- **Batch CSV Classification**: Upload a CSV with galaxy coordinates (RA, Dec) to classify multiple galaxies at once. The app downloads FITS cutouts from Legacy Survey DR10 and runs inference on each.
- **Single FITS Analysis**: Upload a `.fits` file or enter coordinates to get a detailed morphological analysis with ring detection overlay, bar detection, radial profile, and class probabilities.
- **Ring Detection Pipeline**: Full morphological analysis with azimuthal profile validation, isophote ellipse fitting, and bar detection (v5 pipeline from the notebook).

## Supported CSV Formats

**SDSS style:**
```csv
objID,ra,dec,z
1237648721210769659,134.44717,-0.1999727,0.02820569
```

**MaNGA style:**
```csv
name,anillos,objra,objdec,nsa_z
manga-10001-1902,4,134.1939234,56.786747,0.0253909
```

## Local Development

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deploy to Railway

1. Push this folder to a GitHub repository
2. Go to [railway.app](https://railway.app) and create a new project
3. Connect your GitHub repo
4. Railway will auto-detect the config from `railway.toml` and `nixpacks.toml`
5. The app will be live at the provided Railway URL

### Environment Notes

- **Railway** uses CPU by default — the `nixpacks.toml` installs CPU-only PyTorch to save ~2GB
- The model checkpoint (`.pt` file) can be uploaded through the sidebar at runtime
- For production, you can bake the checkpoint into the Docker image or store it in Railway's volume storage

### Alternative: Dockerfile

If you prefer Docker:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

## Model Architecture

- **Encoder**: Zoobot ConvNeXt-nano (`hf_hub:mwalmsley/zoobot-encoder-convnext_nano`)
- **Head**: BatchNorm → Dropout(0.4) → Linear(encoder_dim, 128) → GELU → BatchNorm → Dropout(0.2) → Linear(128, 3)
- **Loss**: FocalLoss (γ=2.0, label_smoothing=0.05)
- **Training**: 3-phase progressive (head only → top encoder layers → full fine-tune)
- **Classes**: `inner`, `outer`, `inner+outer`

## Data Sources

- SDSS DR10 (dataset.csv) — objID, ra, dec, ring code
- MaNGA DR9 (MaNGA_rings.csv) — galaxy name, ring code, coordinates
- Legacy Survey DR10 FITS cutouts (g, r, z bands)
