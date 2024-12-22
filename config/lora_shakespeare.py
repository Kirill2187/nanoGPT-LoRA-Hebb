import time

out_dir = 'out-lora-shakespeare'
eval_interval = 5
eval_iters = 60
wandb_log = True
wandb_project = 'lora-shakespeare'
wandb_run_name = 'bp-' + str(time.time())

dataset = 'shakespeare'
init_from = 'gpt2' # 124M GPT-2

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

batch_size = 8
gradient_accumulation_steps = 4
max_iters = 100000
early_stopping = 5

learning_rate = 1e-4
decay_lr = False

compile = False

lora_rank = 16
lora_alpha = 32
lora_dropout = 0
lora_matrices = "qv"
