from copy import deepcopy
import math
import numpy as np
import os
import random
import time

from dotenv import load_dotenv
import smart_open
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import GPT2Config, get_scheduler
import wandb

import config as cfg
from utils import *

# Set up distributed training
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

if world_size > 1:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl') if world_size > 1 else None
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
# Set random seed
seed = cfg.RANDOM_SEED + global_rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load environment variables from .env file.
# -- Set AWS or other cloud storage keys as env variables here.
# -- Set wandb API key as env variable here.
load_dotenv()

def get_config_vars(module):
    return {key: value for key, value in vars(module).items() if not key.startswith('__')}

wandb.init(
    project="Compression Experiments",
    config=get_config_vars(cfg),
    mode="online" if cfg.LOG_WANDB_ONLINE else "offline",
)


def collate_batch(input_patches):

    input_patches, labels = zip(*input_patches)
    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=256)
    labels = torch.stack(labels, dim=0)

    return input_patches.to(device), labels.to(device)

def list_files_in_directory(directories):
    file_list = []
    
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list

def read_bytes(filename):
    
    ext = filename.split('.')[-1]
    ext = bytearray(ext, 'utf-8')
    ext = [byte for byte in ext][:cfg.PATCH_SIZE]

    with open(filename, 'rb') as f:
        file_bytes = f.read()

    bytes = []
    for byte in file_bytes:
        bytes.append(byte)

    if len(bytes)%cfg.PATCH_SIZE!=0:
        bytes = bytes + [256] * (cfg.PATCH_SIZE - len(bytes) % cfg.PATCH_SIZE)

    bos_patch = ext + [256] * (cfg.PATCH_SIZE - len(ext))
    bytes = bos_patch + bytes + [256] * cfg.PATCH_SIZE

    return bytes

class ByteDataset(Dataset):
    def __init__(self, filenames):
        print(f"Loading {len(filenames)} files for classification")
        self.filenames = []
        self.labels = {}

        for filename in tqdm(filenames):
            file_size = os.path.getsize(filename)
            file_size = math.ceil(file_size / cfg.PATCH_SIZE)
            ext = filename.split('.')[-1]
            label = os.path.basename(filename).split('_')[0]
            label = f"{label}.{ext}"

            if file_size <= cfg.PATCH_LENGTH-2:
                self.filenames.append((filename, label))
                if label not in self.labels:
                    self.labels[label] = len(self.labels)
            
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        filename, label = self.filenames[idx]
        file_bytes = read_bytes(filename)

        file_bytes = torch.tensor(file_bytes, dtype=torch.long)
        label = torch.tensor(self.labels[label], dtype=torch.long)
        
        return file_bytes, label

train_files = list_files_in_directory(cfg.TRAIN_FOLDERS)
eval_files = list_files_in_directory(cfg.EVAL_FOLDERS)

train_set = ByteDataset(train_files)
eval_set = ByteDataset(eval_files)

patch_config = GPT2Config(num_hidden_layers=cfg.PATCH_NUM_LAYERS, 
                    max_length=cfg.PATCH_LENGTH, 
                    max_position_embeddings=cfg.PATCH_LENGTH,
                    hidden_size=cfg.HIDDEN_SIZE,
                    n_head=cfg.HIDDEN_SIZE//64,
                    vocab_size=1)
model = bGPTForClassification(patch_config, len(train_set.labels))
model = model.to(device)

print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=True)

scaler = GradScaler()
is_autocast = True
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# call model with a batch of input
def process_one_batch(batch):
    input_patches, labels = batch
    logits = model(input_patches)
    loss = loss_fn(logits, labels)
    prediction = torch.argmax(logits, dim=1)
    acc_num = torch.sum(prediction==labels)

    return loss, acc_num

# do one epoch for training
def train_epoch():
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    total_acc_num = 0
    iter_idx = 1
    model.train()

    for batch in tqdm_train_set:
        if is_autocast:
            with autocast():
                loss, acc_num = process_one_batch(batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, acc_num = process_one_batch(batch)
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        total_train_loss += loss.item()
        total_acc_num += acc_num.item()
        tqdm_train_set.set_postfix({str(global_rank)+'_train_acc': total_acc_num / (iter_idx*cfg.BATCH_SIZE)})
        if iter_idx % 20 == 0:
            wandb.log({"batch_loss": loss.item()})
        iter_idx += 1
        
    return total_acc_num / ((iter_idx-1)*cfg.BATCH_SIZE)

# do one epoch for eval
def eval_epoch():
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    total_acc_num = 0
    iter_idx = 1
    model.eval()
  
    # Evaluate data for one epoch
    for batch in tqdm_eval_set: 
        with torch.no_grad():
            loss, acc_num = process_one_batch(batch)
            total_eval_loss += loss.item()
            total_acc_num += acc_num.item()
        tqdm_eval_set.set_postfix({str(global_rank)+'_eval_acc': total_acc_num / (iter_idx*cfg.BATCH_SIZE)})
        iter_idx += 1
    wandb.log({"total_eval_loss": total_eval_loss})
    return total_acc_num / ((iter_idx-1)*cfg.BATCH_SIZE)

# train and eval
if __name__ == "__main__":

    labels = train_set.labels

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=global_rank)
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=global_rank)

    train_set = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, collate_fn=collate_batch, sampler=train_sampler, shuffle = (train_sampler is None))
    eval_set = DataLoader(eval_set, batch_size=cfg.BATCH_SIZE, collate_fn=collate_batch, sampler=eval_sampler, shuffle = (train_sampler is None))

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=cfg.NUM_EPOCHS * len(train_set) // 10,
        num_training_steps=cfg.NUM_EPOCHS * len(train_set),
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    
    if cfg.LOAD_FROM_PRETRAINED:
        # Load checkpoint to CPU
        with smart_open.open(cfg.LOAD_WEIGHTS_PATH, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')

        byte_config = GPT2Config(num_hidden_layers=cfg.BYTE_NUM_LAYERS, 
                            max_length=cfg.PATCH_SIZE+1, 
                            max_position_embeddings=cfg.PATCH_SIZE+1,
                            hidden_size=cfg.HIDDEN_SIZE,
                            n_head=cfg.HIDDEN_SIZE//64,
                            vocab_size=256+1)
        pretrained_model = bGPTLMHeadModel(patch_config, byte_config)
        pretrained_model.load_state_dict(checkpoint['model'])

        # Here, model is assumed to be on GPU
        # Load state dict to CPU model first, then move the model to GPU
        if torch.cuda.device_count() > 1:
            # If you have a DataParallel model, you need to load to model.module instead
            cpu_model = deepcopy(model.module)
            cpu_model.patch_level_decoder.load_state_dict(pretrained_model.patch_level_decoder.state_dict())
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            # Load to a CPU clone of the model, then load back
            cpu_model = deepcopy(model)
            cpu_model.patch_level_decoder.load_state_dict(pretrained_model.patch_level_decoder.state_dict())
            model.load_state_dict(cpu_model.state_dict())
        
        try:
            print(f"Successfully Loaded Pretrained Checkpoint at Epoch {checkpoint['epoch']} with Loss {checkpoint['min_eval_loss']}")
        except:
            print(f"Successfully Loaded Pretrained Checkpoint at Epoch {checkpoint['epoch']} with Acc {checkpoint['max_eval_acc']}")

    if cfg.LOAD_FROM_CHECKPOINT:
        # Load checkpoint to CPU. See smart open documentation for specifying s3 paths local paths work transparently.
        with smart_open.open(cfg.LOAD_WEIGHTS_PATH, 'rb') as f:
            checkpoint = torch.load(f, map_location='cpu')
        # Here, model is assumed to be on GPU
        # Load state dict to CPU model first, then move the model to GPU
        if torch.cuda.device_count() > 1:
            # If you have a DataParallel model, you need to load to model.module instead
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            # Load to a CPU clone of the model, then load back
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())

        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_sched'])
        pre_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        max_eval_acc = checkpoint['max_eval_acc']
        labels = checkpoint['labels']
        print("Successfully Loaded Checkpoint from Epoch %d" % pre_epoch)
    
    else:
        pre_epoch = 0
        best_epoch = 0
        max_eval_acc = 0

    for epoch in range(1, cfg.NUM_EPOCHS+1-pre_epoch):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        epoch += pre_epoch
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)
        train_acc = train_epoch()
        eval_acc = eval_epoch()
        wandb.log({"epoch_num": epoch, "epoch_train_acc": train_acc, "epoch_eval_acc": eval_acc})
        if global_rank==0:
            with open(cfg.LOGS_PATH,'a') as f:
                f.write("Epoch " + str(epoch) + "\ntrain_acc: " + str(train_acc) + "\neval_acc: " +str(eval_acc) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")
            if eval_acc > max_eval_acc:
                best_epoch = epoch
                max_eval_acc = eval_acc
                # Checkpoint best epoch.
                checkpoint = { 
                                'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_sched': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                'max_eval_acc': max_eval_acc,
                                "labels": labels
                                }
                checkpoint_name = '{}_best_epoch.pth'.format(cfg.EXPERIMENT_NAME)
                path = os.path.join(cfg.SAVE_WEIGHTS_PATH, checkpoint_name)
                with smart_open(path, 'wb') as f:
                    torch.save(checkpoint, f)
                with open(cfg.LOGS_PATH,'a') as f:
                    f.write("Best Epoch so far!\n")
        # Checkpoint most recent epoch.
        checkpoint = { 
                        'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_sched': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'best_epoch': best_epoch,
                        'max_eval_acc': max_eval_acc,
                        "labels": labels
                        }
        checkpoint_name = '{}_most_recent_epoch.pth'.format(cfg.EXPERIMENT_NAME)
        path = os.path.join(cfg.SAVE_WEIGHTS_PATH, checkpoint_name)
        with smart_open(path, 'wb') as f:
            torch.save(checkpoint, f)
        
        if world_size > 1:
            dist.barrier()

    if global_rank==0:
        print("Best Eval Epoch : "+str(best_epoch))
        print("Max Eval Accuracy : "+str(max_eval_acc))
    
    wandb.finish()