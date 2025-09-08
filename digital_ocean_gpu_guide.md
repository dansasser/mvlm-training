# Digital Ocean GPU Instance Setup Guide for MVLM Training

This guide will walk you through setting up a Digital Ocean GPU instance to train your MVLM (Minimum Viable Language Model).

## 🎯 Overview

You'll create a GPU-enabled droplet on Digital Ocean, set up the training environment, upload your dataset, and train the MVLM. Total estimated cost: **$6-16** for complete training.

## 💰 Cost Breakdown

### GPU Instance Options:
- **H100 (Recommended):** $7.20/hour - Training time: 1-2 hours = **$7-14 total**
- **A100:** $3.60/hour - Training time: 2-3 hours = **$7-11 total**  
- **V100:** $1.20/hour - Training time: 4-6 hours = **$5-7 total**

### Additional Costs:
- **Storage:** ~$1/month for 50GB SSD
- **Bandwidth:** Minimal for dataset upload/download

## 🚀 Step-by-Step Setup

### Step 1: Create Digital Ocean Account
1. Go to [digitalocean.com](https://digitalocean.com)
2. Sign up for an account
3. Add payment method
4. Verify your account

### Step 2: Create GPU Droplet
1. **Click "Create" → "Droplets"**
2. **Choose Region:** Select closest to your location
3. **Choose Image:** Ubuntu 22.04 LTS x64
4. **Choose Size:** 
   - Click "GPU Droplets" tab
   - Select **H100-1x** (recommended) or **A100-1x**
5. **Authentication:** 
   - Choose "SSH Key" (recommended) or "Password"
   - If SSH: Upload your public key or create new one
6. **Hostname:** `mvlm-training-server`
7. **Click "Create Droplet"**

### Step 3: Connect to Your Droplet
```bash
# SSH connection (replace with your droplet IP)
ssh root@your-droplet-ip

# Or if using password, you'll be prompted
```

### Step 4: Run Setup Script
```bash
# Download and run the setup script
curl -O https://your-setup-script-url/digital_ocean_setup.sh
chmod +x digital_ocean_setup.sh
./digital_ocean_setup.sh
```

**Note:** The setup script will:
- Install NVIDIA drivers and CUDA
- Set up Python environment
- Install PyTorch and dependencies
- Create project structure
- Configure training scripts

### Step 5: Upload Training Dataset
```bash
# From your local machine, upload the dataset
scp mvlm_training_dataset_complete.tar.gz root@your-droplet-ip:~/mvlm_training/data/

# On the server, extract the dataset
cd ~/mvlm_training/data
tar -xzf mvlm_training_dataset_complete.tar.gz
```

### Step 6: Start Training
```bash
# Navigate to project directory
cd ~/mvlm_training

# Activate environment and start training
./train_mvlm.sh
```

### Step 7: Monitor Training
```bash
# In a new terminal/session, monitor progress
./monitor_training.sh

# Or watch GPU usage
watch -n 1 nvidia-smi
```

### Step 8: Download Trained Model
```bash
# After training completes, download the model
scp -r root@your-droplet-ip:~/mvlm_training/outputs/ ./mvlm_trained_model/
```

### Step 9: Destroy Droplet (Save Money!)
1. Go to Digital Ocean dashboard
2. Select your droplet
3. Click "Destroy"
4. Confirm destruction

**Important:** Don't forget this step to avoid ongoing charges!

## 📊 Expected Training Timeline

### H100 Instance (Recommended):
- **Setup:** 10-15 minutes
- **Data Upload:** 5-10 minutes  
- **Training:** 1-2 hours
- **Download:** 5-10 minutes
- **Total Time:** 2-3 hours
- **Total Cost:** $7-14

### Training Progress Indicators:
- **Initial setup:** Model architecture creation
- **Data loading:** Dataset tokenization and preparation
- **Training epochs:** 3 epochs with progress bars
- **Evaluation:** Perplexity and loss metrics
- **Sample generation:** Test outputs during training
- **Final save:** Model and tokenizer saved

## 🔧 Troubleshooting

### Common Issues:

#### 1. CUDA Not Available
```bash
# Check NVIDIA drivers
nvidia-smi

# If not working, reboot and try again
sudo reboot
```

#### 2. Out of Memory
```bash
# Reduce batch size in training script
python mvlm_trainer.py --batch_size 4
# Or use gradient accumulation
python mvlm_trainer.py --batch_size 4 --gradient_accumulation_steps 2
```

#### 3. Dataset Not Found
```bash
# Check data directory structure
ls -la data/mvlm_comprehensive_dataset/
```

#### 4. Training Stuck
```bash
# Check training logs
tail -f mvlm_training.log
```

### Getting Help:
- Check `system_info.sh` for system status
- Review `mvlm_training.log` for detailed logs
- Monitor GPU with `nvidia-smi`
- Check disk space with `df -h`

## 📁 File Structure After Setup

```
~/mvlm_training/
├── mvlm_env/                 # Python virtual environment
├── data/
│   └── mvlm_comprehensive_dataset/  # Your training data
├── outputs/
│   ├── mvlm_final/          # Trained model (after training)
│   ├── training_plots.png   # Training visualizations
│   └── checkpoints/         # Training checkpoints
├── mvlm_trainer.py          # Main training script
├── train_mvlm.sh            # Training wrapper
├── monitor_training.sh      # Progress monitor
└── mvlm_training.log        # Training logs
```

## 🎯 Success Indicators

### Training Started Successfully:
```
[INFO] Starting MVLM training...
[INFO] Dataset size: 15000+ examples
[INFO] Using GPU: NVIDIA H100
[INFO] Training for 3 epochs
```

### Training Progressing:
```
Epoch 1, Batch 100/1500, Loss: 3.245, LR: 5.00e-05
Epoch 1, Batch 200/1500, Loss: 2.987, LR: 5.00e-05
```

### Training Complete:
```
Training completed in 3600 seconds (60 minutes)
Final evaluation loss: 1.234
Final perplexity: 3.43
Model saved to: outputs/mvlm_final
```

## 🚀 Next Steps After Training

1. **Download your trained model**
2. **Destroy the Digital Ocean droplet** (important!)
3. **Integrate MVLM with SIM-ONE Framework**
4. **Test the complete system**
5. **Deploy for production use**

## 💡 Pro Tips

### Cost Optimization:
- Use **H100** for fastest training (lowest total cost)
- **Destroy droplet immediately** after downloading model
- **Monitor training** to catch issues early
- **Use snapshots** if you need to pause/resume

### Performance Optimization:
- **Upload dataset first** before starting expensive GPU time
- **Test with small batch** to verify setup
- **Monitor GPU utilization** (should be 80-95%)
- **Save checkpoints** in case of interruption

### Security:
- **Use SSH keys** instead of passwords
- **Change default ports** if keeping droplet longer
- **Enable firewall** for extended use
- **Regular backups** of important data

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review training logs: `cat mvlm_training.log`
3. Check system status: `./system_info.sh`
4. Monitor resources: `htop` and `nvidia-smi`

**Remember:** The goal is to train the MVLM quickly and cost-effectively. Don't leave the GPU instance running unnecessarily!

---

**Ready to start? Create your Digital Ocean account and follow the steps above. Your biblically-grounded MVLM will be ready in just a few hours!** 🎉

