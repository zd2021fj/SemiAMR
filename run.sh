# SSLFMM2 有标签 1% 0dB

#python main-SSL.py --dataset_name=10a --program_seed=0 --lr=0.01 --snr=0 --label_rate=0.01 --lsp_batch_size=64 --usp_batch_size=64 --arch=sslnet --model=SemiAMR --epochs=10 --version=1 --optim=adam --lr_scheduler=cos --weight_rampup=30 --fmix_alpha=1.0 --ema_decay=0.99 --usp_weight=30.0 --workers=16 --save_freq=100 2>&1 | tee results/ours/10a/10a_sslfmm2_0dB_$(date +%y-%m-%d-%H-%M).txt

python main-SSL.py --dataset_name=10b --program_seed=0 --lr=0.01 --snr=0 --label_rate=0.01 --lsp_batch_size=64 --usp_batch_size=64 --arch=sslnet --model=SemiAMR --epochs=10 --version=1 --optim=adam --lr_scheduler=cos --weight_rampup=30 --fmix_alpha=1.0 --ema_decay=0.99 --usp_weight=30.0 --workers=16 --save_freq=100 2>&1 | tee results/ours/10a/10b_sslfmm2_0dB_$(date +%y-%m-%d-%H-%M).txt

python main-SSL.py --dataset_name=04c --program_seed=0 --lr=0.01 --snr=0 --label_rate=0.01 --lsp_batch_size=64 --usp_batch_size=64 --arch=sslnet --model=SemiAMR --epochs=10 --version=1 --optim=adam --lr_scheduler=cos --weight_rampup=30 --fmix_alpha=1.0 --ema_decay=0.99 --usp_weight=30.0 --workers=16 --save_freq=100 2>&1 | tee results/ours/10a/04c_sslfmm2_0dB_$(date +%y-%m-%d-%H-%M).txt
