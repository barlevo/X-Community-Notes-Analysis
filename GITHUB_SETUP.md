# GitHub Setup Guide

Follow these steps to upload your project to GitHub.

## Step 1: Create a GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **+** icon in the top right → **New repository**
3. Fill in:
   - **Repository name**: `x-community-notes-analysis` (or your preferred name)
   - **Description**: "Data Science project analyzing X Community Notes using topic classification and community detection"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **Create repository**

## Step 2: Initialize Git (if not already done)

Open terminal/command prompt in your project directory and run:

```bash
# Check if git is already initialized
git status

# If not initialized, run:
git init
```

## Step 3: Add and Commit Files

```bash
# Add all files (respects .gitignore)
git add .

# Check what will be committed (optional)
git status

# Commit with a message
git commit -m "Initial commit: X Community Notes analysis project"
```

## Step 4: Connect to GitHub and Push

```bash
# Add GitHub remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Rename main branch if needed (GitHub uses 'main' by default)
git branch -M main

# Push to GitHub
git push -u origin main
```

**Note**: You'll be prompted for your GitHub username and password (or personal access token).

## Alternative: Using GitHub Desktop

If you prefer a GUI:

1. Download [GitHub Desktop](https://desktop.github.com/)
2. Sign in with your GitHub account
3. File → Add Local Repository → Select your project folder
4. Click "Publish repository" button
5. Choose name and visibility
6. Click "Publish repository"

## What Gets Uploaded

✅ **Will be uploaded:**
- All Python code files
- README.md
- requirements.txt
- Documentation files
- Project structure

❌ **Will NOT be uploaded** (thanks to .gitignore):
- Data files (data/ directory)
- Model files (*.pkl)
- Generated results and plots
- Virtual environment
- Cache files

## Important Notes

1. **Data Files**: Your actual data files (TSV files) are NOT uploaded. This is intentional - they're too large and should be kept private.

2. **Personal Access Token**: If you use 2FA on GitHub, you'll need a Personal Access Token instead of a password:
   - GitHub → Settings → Developer settings → Personal access tokens → Generate new token
   - Use this token as your password when pushing

3. **Future Updates**: To update your repository:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push
   ```

## Quick Commands Reference

```bash
# Check status
git status

# Add files
git add .

# Commit
git commit -m "Your commit message"

# Push to GitHub
git push

# Pull latest changes (if working on multiple machines)
git pull
```

---

**Your project is now on GitHub!** 🎉

Share the repository URL with others or add it to your portfolio/resume.

