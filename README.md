# WasteHub API - Deployment Package

This folder contains only the files needed for deployment.

## Files Included

- `wastehub_api.py` - Main FastAPI application
- `requirements.txt` - Python dependencies
- `hazard_type_classes.json` - Class labels
- Model files (`.h5`) - Trained ML models
- Deployment configs (Procfile, runtime.txt, etc.)

## Quick Deploy to Render

1. **Initialize Git:**
   ```bash
   git init
   git add .
   git commit -m "WasteHub API deployment"
   ```

2. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/wastehub-api.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy on Render:**
   - Go to [render.com](https://render.com)
   - Connect GitHub repository
   - Select this repository
   - Deploy!

## Model Files

If model files are large (>100MB), consider using Git LFS:
```bash
git lfs install
git lfs track "*.h5"
git add .gitattributes
git add .
git commit -m "Add models with Git LFS"
git push
```

## Testing

After deployment, test your API:
```bash
curl https://your-app.onrender.com/
```
