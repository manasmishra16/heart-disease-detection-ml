# 🚀 How Friends Can See Your Project

## Option 1: View on GitHub 🌐
**Simplest way - Just share the link!**
```
https://github.com/manasmishra16/heart-disease-detection-ml
```
Your friend can browse code, read documentation, and download the project.

---

## Option 2: Share Live Demo URL 🌍
**Deploy online so anyone can access via browser**

### Deploy to Streamlit Community Cloud (FREE)

1. **Go to:** https://share.streamlit.io/

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Fill in details:**
   - Repository: `manasmishra16/heart-disease-detection-ml`
   - Branch: `main`
   - Main file path: `app_final.py`

5. **Click "Deploy"**

6. **Share the URL** with your friend (e.g., `https://your-app.streamlit.app`)

⚠️ **Important Notes:**
- You'll need to train models first (models are in .gitignore)
- Either: 
  - Remove model files from .gitignore and push them
  - OR create a startup script to download/train models on deployment

---

## Option 3: Run Locally 💻
**If your friend has Python installed**

### Step 1: Clone Repository
```bash
git clone https://github.com/manasmishra16/heart-disease-detection-ml.git
cd heart-disease-detection-ml
```

### Step 2: Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### Step 3: Train Models (Required)
```bash
python train_best_model.py
```

### Step 4: Run Demo
```bash
streamlit run app_final.py
```

### Step 5: Open Browser
Navigate to: `http://localhost:8501`

---

## Option 4: Share Screen/Video 📹
**Quick demo without setup**

1. **Run locally:**
   ```bash
   streamlit run app_final.py --server.port 8506
   ```

2. **Record screen** using:
   - Windows: Xbox Game Bar (Win + G)
   - OBS Studio
   - Loom (browser-based)

3. **Share video** via:
   - YouTube (unlisted)
   - Google Drive
   - Discord/WhatsApp

---

## Option 5: Port Forwarding (Advanced) 🔌
**Share your local server temporarily**

### Using ngrok (FREE):

1. **Download ngrok:** https://ngrok.com/download

2. **Run your app:**
   ```bash
   streamlit run app_final.py --server.port 8506
   ```

3. **In another terminal:**
   ```bash
   ngrok http 8506
   ```

4. **Share the URL** (e.g., `https://abc123.ngrok.io`)

⚠️ **Security Note:** Only share with trusted friends, expires when you close ngrok.

---

## 🎯 Recommended Approach

**For Non-Technical Friends:**
→ Deploy to Streamlit Cloud (Option 2) or share video (Option 4)

**For Technical Friends:**
→ Share GitHub link (Option 1) and let them run locally (Option 3)

**For Quick Demo:**
→ Use ngrok (Option 5) or screen share

---

## 📝 Before Sharing

✅ **Checklist:**
- [ ] Update README.md with project overview
- [ ] Add screenshots to docs/
- [ ] Test app_final.py works correctly
- [ ] Ensure requirements_streamlit.txt is complete
- [ ] Add deployment instructions
- [ ] Consider adding model files or training script

---

## 🆘 Troubleshooting

**Models not found error?**
- Models are in .gitignore
- Either push models or run `python train_best_model.py` first

**Dependencies missing?**
- Use `requirements_streamlit.txt` for deployment
- Run: `pip install -r requirements_streamlit.txt`

**Port already in use?**
- Change port: `streamlit run app_final.py --server.port 8507`

---

## 📞 Support

If your friend has issues:
1. Check GitHub Issues tab
2. Refer to QUICK_START.md
3. Contact you directly

**Live Demo URL (after deployment):**
`[Add your Streamlit Cloud URL here]`
