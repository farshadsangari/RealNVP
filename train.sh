# Image channels
input_channels=3


# RealNVP parameters
num_scales=3
middle_channels=64


# Training parameters
num_epochs=100
batch_size=64   
lr=1e-3
weight_decay=5e-5
max_grad_norm=100
data_constraint=0.9


# Saving parameters
ckpt_save_path='./ckpts'
ckpt_prefix='cktp_epoch_'
ckpt_save_freq=10
report_path="./reports"
base_raw_data_path="./data/raw_data/img_align_celeba/img_align_celeba"


# Loss(objective function) parameters
distribution_mean=0
distribution_std=1
possible_values_in_each_input_dimension=256



python train.py --num-input-channels $input_channels --num-scales $num_scales \
                --batch-size $batch_size --num-middle-channels $middle_channels \
                --lr $lr --weight-decay $weight_decay --num-epochs $num_epochs \
                --max-grad-norm $max_grad_norm --ckpt-save-path $ckpt_save_path \
                --ckpt-prefix $ckpt_prefix --ckpt-save-freq $ckpt_save_freq \
                --distribution-mean $distribution_mean --distribution-std $distribution_std \
                --possible-values-in-each-input-dimension $possible_values_in_each_input_dimension \
                --data-constraint $data_constraint --report-path $report_path --base-raw-data-path $base_raw_data_path       
