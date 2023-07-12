export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="data/dog_db"
export OUTPUT_DIR="outputs/dog_db"

accelerate launch main.py --exp_name=train \
    --train.pretrained_model_name_or_path=$MODEL_NAME  \
    --train.instance_data_dir=$INSTANCE_DIR \
    --train.output_dir=$OUTPUT_DIR \
    --train.instance_prompt="a photo of sks dog" \
    --train.resolution=512 \
    --train.train_batch_size=1 \
    --train.gradient_accumulation_steps=1 \
    --train.learning_rate=5e-6 \
    --train.lr_scheduler="constant" \
    --train.lr_warmup_steps=0 \
    --train.max_train_steps=400