# 🚀 Deploy to Railway - Step by Step Guide

## Prerequisites
- GitHub account
- Railway account (sign up at https://railway.app)
- Dhan API credentials (Client ID and Access Token)

## Step 1: Push Code to GitHub

```bash
# Create new repository on GitHub (via web interface)
# Then add remote and push

git remote add origin https://github.com/YOUR_USERNAME/dhan-scanner.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy to Railway

### Option A: Deploy via Railway Dashboard (Recommended)

1. **Login to Railway**: https://railway.app

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Connect your GitHub account if not already connected
   - Select your `dhan-scanner` repository

3. **Configure Environment Variables**:
   - Click on your deployment
   - Go to "Variables" tab
   - Add the following variables:
   ```
   DHAN_CLIENT_ID=your_actual_client_id
   DHAN_ACCESS_TOKEN=your_actual_access_token
   TELEGRAM_TOKEN=your_bot_token (optional)
   TELEGRAM_CHAT_ID=your_chat_id (optional)
   ```

4. **Deploy**:
   - Railway will automatically detect the Procfile and start deployment
   - Wait for deployment to complete (2-3 minutes)
   - Your app will be available at: `https://your-app-name.railway.app`

### Option B: Deploy via Railway CLI

1. **Install Railway CLI**:
   ```bash
   # Windows (PowerShell as Admin)
   npm install -g @railway/cli

   # Or using curl
   curl -fsSL https://railway.app/install.sh | sh
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Create New Project**:
   ```bash
   railway new dhan-scanner
   ```

4. **Set Environment Variables**:
   ```bash
   railway variables set DHAN_CLIENT_ID=your_actual_client_id
   railway variables set DHAN_ACCESS_TOKEN=your_actual_access_token
   railway variables set TELEGRAM_TOKEN=your_bot_token
   railway variables set TELEGRAM_CHAT_ID=your_chat_id
   ```

5. **Deploy**:
   ```bash
   railway up
   ```

6. **Get Your App URL**:
   ```bash
   railway open
   ```

## Step 3: Verify Deployment

1. **Check Deployment Status**:
   - Go to Railway dashboard
   - Click on your project
   - Check the "Deployments" tab for logs

2. **Access Your App**:
   - Open: `https://your-app-name.railway.app`
   - You should see the F&O Scanner Dashboard

3. **Check Health Endpoint**:
   ```bash
   curl https://your-app-name.railway.app/api/status
   ```

## Step 4: Configure Custom Domain (Optional)

1. In Railway dashboard, go to Settings
2. Add your custom domain
3. Update DNS records as instructed

## Important Notes

### Security
- ✅ Never commit `.env` files with real credentials
- ✅ Always use Railway's environment variables for secrets
- ✅ The scanner will run in demo mode if credentials are missing

### Railway Limits (Free Tier)
- 500 hours/month execution time
- 100GB bandwidth
- Sleeps after 30 minutes of inactivity
- Consider upgrading for production use

### Monitoring
- View logs: Railway Dashboard → Deployments → View Logs
- Set up alerts: Configure Telegram notifications
- Database: SQLite data persists across deployments

### Troubleshooting

**App not starting?**
- Check logs in Railway dashboard
- Verify environment variables are set correctly
- Ensure all required files are committed

**WebSocket connection issues?**
- Railway supports WebSockets by default
- Check if Dhan credentials are correct
- Verify network connectivity

**Database errors?**
- SQLite file is created automatically
- Data persists in Railway's volume

## Step 5: Production Recommendations

1. **Upgrade Railway Plan** for:
   - No sleep/always-on
   - More resources
   - Custom domains

2. **Add Monitoring**:
   - Set up Telegram alerts
   - Configure health checks
   - Add error tracking (Sentry)

3. **Scale Configuration**:
   - Start with `batch_size: 50` in config.json
   - Gradually increase based on performance
   - Monitor memory usage in Railway metrics

## Quick Commands Reference

```bash
# View logs
railway logs

# Restart deployment
railway restart

# Update environment variable
railway variables set KEY=value

# Open app in browser
railway open

# Check status
railway status
```

## Support

- Railway Documentation: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- GitHub Issues: Report bugs in your repository

---

🎉 **Congratulations!** Your Dhan F&O Scanner is now live on Railway!

Access your dashboard at: `https://your-app-name.railway.app`