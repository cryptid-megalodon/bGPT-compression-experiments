# Configuration for generative modelling and classification
TRAIN_FOLDERS = [
    "data/wikipedia/uncompressed/train",
    # "data/ag_news/uncompressed/train",
    # "imagenet32/train",
    # "cifar/train",
    # "librispeech8K/train",
    # "speech_commands8K/train",
    # "irishman/train",
    # "cpu_states/train",
]  # Folder containing training data
EVAL_FOLDERS = [
    "data/wikipedia/uncompressed/test",
    # "data/ag_news/uncompressed/test",
    # "imagenet32/test",
    # "cifar/test",
    # "librispeech8K/test",
    # "speech_commands8K/test",
    # "irishman/test",
    # "cpu_states/test",
]  # Folder containing evaluation data

# Configuration for the paths
LOAD_WEIGHTS_PATH = "s3://cryptid-megalodon-compression-experiments/weights-text.pth"  # Path to save weights
SAVE_WEIGHTS_DIR_PATH = (
    "s3://cryptid-megalodon-compression-experiments"
    # BUCKET_PATH  # The checkpoint will be the experiment name plus the wandb run ID.
)
LOGS_PATH = "logs-test-ag-cls.txt"  # Path to save logs

# Configuration for the model
PATCH_SIZE = 16  # Patch Size
PATCH_LENGTH = 512  # Patch Length
BYTE_NUM_LAYERS = 3  # Number of layers in the decoder
PATCH_NUM_LAYERS = 12  # Number of layers in the encoder
HIDDEN_SIZE = 768  # Hidden Size
COMPRESS_BYTES = True  # Compress the content of the file before inference.

# Configuration for the training
EXPERIMENT_NAME = "compression-pretraining-text"
LOG_WANDB_ONLINE = True  # Log wandb to online server. If true, you will need to provide your wandb API key as an environment variable when running the docker container.
RANDOM_SEED = 0  # Controls the random number seed for all RNGs in the experiment.
NUM_EPOCHS = 32  # Number of epochs to train for (if early stopping doesn't intervene)
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
BATCH_SIZE = 16  # Batch size for training
ACCUMULATION_STEPS = 1  # Accumulation steps to simulate large batch size
PATCH_SAMPLING_BATCH_SIZE = (
    0  # Batch size for patch during training, 0 for full conaudio
)
LOAD_FROM_CHECKPOINT = False  # Whether to load weights from a checkpoint
LOAD_FROM_PRETRAINED = False  # Whether to load pre-trained weights from a checkpoint
SAVE_FINAL_MODEL = True  # Whether to save the final checkpoint or not.
CONVERSION_MODE = None  # Mode of conversion (None for regular training, input->output for unidirectional conversion, input&output for bidirectional conversion)

# Configuration for inference
INFERENCE_WEIGHTS_PATH = "weights-conversion.pth"  # Path to weights for inference
INPUT_EXT = "abc"  # Extension of input files, used for conversion
TARGET_EXT = "mid"  # Extension of target files
INPUT_FOLDER = "input"  # Folder containing input files
OUTPUT_FOLDER = "output"  # Folder to save output files
MODE = "convert"  # Mode of inference (convert or generate)
NUM_SAMPLES = 100  # Number of samples to generate (only for generate mode)
TOP_K = 0  # Top k for sampling
TOP_P = 1.0  # Top p for sampling
TEMPERATURE = 1  # Temperature for sampling
