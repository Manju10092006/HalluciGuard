# 🚀 Quick Hugging Face Space Deployment Guide

Follow these steps to deploy this HalluciGuard Detector backend to Hugging Face Spaces in under 2 minutes:

---

### Step 1: Create a New Space
1. Log into your Hugging Face account.
2. Click on your profile icon (top right) and click **"New Space"** (or visit: https://huggingface.co/new-space).
3. Configure the Space settings:
   - **Space Name**: `halluciguard-detector` (or any name you choose)
   - **License**: `MIT`
   - **Space SDK**: Select **`Gradio`** *(Important: Do NOT select Docker or Streamlit)*
   - **Space Hardware**: Select **`CPU basic`** (Free · 2 vCPU · 16 GB RAM)
   - **Space Visibility**: **`Public`**
4. Click **"Create Space"**.

---

### Step 2: Upload the Files
You can upload either via your browser or via Git:

#### Method A: Direct Browser Upload (Easiest)
1. In your newly created Space, click on the **"Files"** tab.
2. Click **"Add file"** → **"Upload files"**.
3. Drag and drop all the contents of this folder into the upload box:
   - `README.md`
   - `gradio_app.py`
   - `app.py`
   - `requirements.txt`
   - `.env.example`
   - The entire `detector/` folder
   - The entire `tests/` folder
4. At the bottom, click **"Commit changes to main"**.

#### Method B: Git Push (Alternative)
```bash
git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
cd <your-space-name>
# Copy all files from this zip folder into the cloned directory
git add .
git commit -m "Deploy HalluciGuard Detector Agent"
git push
```

---

### Step 3: Verify It Is Running
1. Hugging Face will automatically begin building the container.
2. In about 1 to 2 minutes, the Space status will change to **"Running"**.
3. You can test the detector immediately on the interactive Gradio UI right on the Space page!

---

### 🌐 Live API Endpoints for Frontend Integration
Once running, the Space provides:
- **Interactive UI**: `https://<your-username>-<space-name>.hf.space`
- **Detection API**: `POST https://<your-username>-<space-name>.hf.space/detect`
- **Health Check**: `GET https://<your-username>-<space-name>.hf.space/health`
- **Model Info**: `GET https://<your-username>-<space-name>.hf.space/model-info`

> **Note on Model:** The application automatically loads the fine-tuned classifier weights directly from the public Hugging Face repository `Manjunath2000006/halluciguard-detector`. No extra setup or API keys are required.
