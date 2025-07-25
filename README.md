# Byte-based LLMs on Compressed Data

This repository is a fork of the original [bGPT repo](https://github.com/byte-gpt/bGPT).
The original repo was created by [Sander Wood](https://github.com/sanderwood).
Please see the published paper for more details: [Beyond Language Models: Byte Models are Digital World Simulators](https://arxiv.org/abs/2402.19155).
Their original README is reproduced in later sections of this file and its original MIT
license has been retained.

## Introduction
I was impressed by the results of byte-based Transformers. Transformers predicting byte 
sequences are a powerful multi-modal approach that could be applied across most digital
formats. There are interesting applications for this approach including: transformation
of data across file formats, generalized learning across file types, and an alternative
approach to specialized token processing for multi-modal models. The chief drawback of byte-based
transformers is the greatly expanded input sequence length. This is due to the longer
byte string sequence length required to represent data.

## Compressed Data Hypothesis
I had an admittedly outlandish idea to attempt to fine-tune a byte-based transformer on
compressed data. Transformers are remarkably robust to learning data patterns, and my
hypothesis was that certain compressed data formats could also be predicted. Data formats
like zip, which use string lookup tables and offer lossless compression of all data types,
seemed like reasonable targets to attend. The lookup table can be thought of as a
Python dictionary where a single ID references a common sequence found throughout the
file. This enables zip to replace that sequence with a single identifier in the lookup
table. I believed the attention mechanism could easily learn the lookup table pattern
and process the zipped data while greatly reducing the input sequence length of longer
files, leading to more efficient training and inference. This would ameliorate the
chief drawback of byte-based transformers.

### Information Theory Disagrees with my hypothesis!
I wanted to point out that later investigations into how the ZIP Deflate algorithm works would
throw a bucket of cold water on my hypothesis. The Deflate algorithm uses a near entropy
optimal compression algorithm meaning the sequence of bytes would look mostly like random
noise to the model. Interestingly, some researchers have thrown some pretty big transformers
at this problem in an attempt to discover a model that might be the basis for an even more
efficient compression algorithm. They were not succcessful. I didn't fully appreciate this
until I better understood how the compression algorithm works, but I was having a lot of fun
with the project and learning a lot. I decided to finish a few runs and verify I didn't have
compression breakthrough on my hands :).

## Personal Project Goals
Aside from the theoretical basis for the project, I thought this would be a good way to
learn more about ML research since the bGPT model was relatively small and I could easily
fine-tune it on compressed data that I prepared (e.g. zipped copies of Wikipedia and news
article datasets). I could focus on practical implementation details I wanted to practice
like working with research code bases, integrating with wandb, conda and other infra tools
like launching training jobs with docker on neoclouds.

## Results
I was able to train a byte-based transformer on a zipped copy of Wikipedia and a zipped
copy of news articles; however, training would always collapse. Fine-tuning alone was
unable to learn the compression algorithm. I considered following the author's original
pre-training approach but estimated it would cost ~$500 in GPU time. The WandB charts below
show the training loss and eval accuracy for the smaller news datasets between the uncompressed
and compressed fine-tuning runs. These charts were typical of the other 13 parameter sweeps runs
I tried with the larger wikipedia dataset.

<image src=wandb_graphs/loss.png width=500>
<image src=wandb_graphs/accuracy.png width=500>

Through working on this project, I discovered some fundamental flaws in this approach.
While there may be benefits to one-time compression of a large training set, all input
would also need to undergo compression before inference, leading to increased
computation cost and latency. Also, during the course of this training algorithmic advances
like mamba-based input layer architectures (e.g. Jamba) were announced. I was convinced that
the cost of input compression would not exceed architectural reductions to input sequece costs.
Advancements in tokenization may make it unnecessary to compress data before processing.

The allure of the simplicity and universality of byte-based transformers is still
appealing, but I think other approaches to handling long input sequence lenghts are
more promising. I did accomplish my learning goals and feel more confident in my ML
research skills beyond my personal studies and class work.

=========================== Start of the original README ===================================

# Beyond Language Models: Byte Models are Digital World Simulators

This repository contains the code for the bGPT model as described in the paper [Beyond Language Models: Byte Models are Digital World Simulators](https://arxiv.org/abs/2402.19155).

bGPT supports generative modelling via next byte prediction on any type of data and can perform any task executable on a computer, showcasing the capability to simulate all activities within the digital world, with its potential only limited by computational resources and our imagination.

You can check out the [demo page](https://byte-gpt.github.io/), which includes examples generated by the bGPT model.

## Model Description
Traditional deep learning often overlooks bytes, the basic units of the digital world, where all forms of information and operations are encoded and manipulated in binary format. Inspired by the success of next token prediction in natural language processing, we introduce bGPT, a model with next byte prediction to simulate the digital world. bGPT matches specialized models in performance across various modalities, including text, audio, and images, and offers new possibilities for predicting, simulating, and diagnosing algorithm or hardware behaviour. It has almost flawlessly replicated the process of converting symbolic music data, achieving a low error rate of 0.0011 bits per byte in converting ABC notation to MIDI format. In addition, bGPT demonstrates exceptional capabilities in simulating CPU behaviour, with an accuracy exceeding 99.99% in executing various operations. Leveraging next byte prediction, models like bGPT can directly learn from vast binary data, effectively simulating the intricate patterns of the digital world.

We provide five weights of bGPT on [Hugging Face](https://huggingface.co/sander-wood/bgpt/tree/main) corresponding to each dataset used for pre-training:

1. **_weights-conversion.pth_**: bGPT pre-trained on IrishMAN for data conversion (between `.abc` and `.mid`).
2. **_weights-cpu.pth_**: bGPT pre-trained on CPU states for CPU state modelling (`.bin`).
3. **_weights-text.pth_**: bGPT pre-trained on Wikipedia for text generation/classification (`.txt`).
4. **_weights-image.pth_**: bGPT pre-trained on ImageNet for image generation/classification (`.bmp`).
5. **_weights-audio.pth_**: bGPT pre-trained on Librispeech for audio generation/classification (`.wav`).

The core components of bGPT include a 12-layer patch-level decoder, a 3-layer byte-level decoder, with a hidden size of 768, totaling 110 million parameters.

## Installation

To set up the bGPT environment and install the necessary dependencies, follow these steps:

1. **Create and Activate Conda Environment**

   ```bash
   conda create --name bgpt python=3.7.9
   conda activate bgpt
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Pytorch**

   ```bash
   pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
   ```
4. **Download Pre-trained bGPT Weights (Optional)**
   
   For those interested in starting with pre-trained models, bGPT weights are available on [Hugging Face](https://huggingface.co/sander-wood/bgpt/tree/main). This step is optional but recommended for users looking to leverage the model's capabilities without training from scratch.
   
## Usage

- `config.py`: Configuration settings for training and inference.
- `cpu-simulation.py`: Simulate CPU states and operations.
- `inference.py`: Perform inference tasks (e.g., generation and conversion) using pre-trained models.
- `train-cls.py`: Training script for classification models.
- `train-gen.py`: Training script for generative models.
- `utils.py`: Utility functions supporting model operations and data processing.
  
### Configuration
The `config.py` file contains critical settings for training and inference, allowing for flexibility across different tasks and modalities. Here's a breakdown of the main configurations:

#### Training Configuration

- **TRAIN_FOLDERS**: Specify the dataset folders for training. Multiple folders can be included.
- **EVAL_FOLDERS**: Specify evaluation dataset folders.
- **PRETRAINED_PATH**: Path to pre-trained weights for transfer learning and fine-tuning.
- **WEIGHTS_PATH & LOGS_PATH**: Define locations to save trained weights and logs, respectively.
- **NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE**: Control training duration, learning rate, and batch size for optimal learning.
- **ACCUMULATION_STEPS**: Set accumulation steps to emulate larger batch sizes, managing memory usage efficiently.
- **PATCH_SAMPLING_BATCH_SIZE**: Adjust batch size for patch sampling during training to reduce computational load, with `0` for full batch processing.

#### Inference Configuration

- **INFERENCE_WEIGHTS_PATH**: Path to weights for inference.
- **MODE**: Determines operation mode (`convert` or `generate`), guiding the model for specific outcomes.
- **NUM_SAMPLES, TOP_K, TOP_P, TEMPERATURE**: Set sampling strategy during inference to control the diversity of outputs.

### Generative Modelling

Generative modelling with bGPT is a flexible and powerful approach to learning and generating new data across various formats. bGPT segments byte sequences into patches, predicts next patch features with a patch-level decoder, and reconstructs bytes within patches using these features with a byte-level decoder. Here's how to get started:

1. **Prepare Your Data**: bGPT can handle any computer file type, including text, images, audio, executables, and encrypted or proprietary formats, without needing specific adjustments for each modality. This capability allows for straightforward and versatile training on a wide array of data. The only thing you need to do here is simply to split your data for training and evaluation.

2. **Adjust Configuration Settings**: Modify the `config.py` file to tailor the training process to your needs. At a minimum, you should update the `TRAIN_FOLDERS` and `EVAL_FOLDERS` to point to your actual data directories. Also, specify where to save the trained model weights and logs by setting `WEIGHTS_PATH` and `LOGS_PATH`. You may adjust other parameters based on your specific requirements. For instance, with the default `PATCH_SIZE=16` and `PATCH_LENGTH=512`, bGPT can model byte sequences up to 8KB. If your training files are larger, and you have sufficient computational resources, consider increasing these parameters to accommodate the larger file sizes.

3. **Leverage Pre-trained Weights (Optional)**: If you wish to fine-tune a pre-trained bGPT model, set `PRETRAINED_PATH` to the location of the pre-trained weights and ensure `LOAD_FROM_PRETRAINED=True`. To train a model from scratch, simply set `LOAD_FROM_PRETRAINED=False`.

4. **Start Training**: Run `train-gen.py` to begin the training process. The script will use the configurations set in `config.py` and apply the training data to learn generative models capable of producing new, unseen outputs in the format of your training data.

### Data Conversion

The conversion mode in bGPT adds a specialized functionality for transforming data from one format to another, leveraging the model's understanding of byte sequences across different file types. This mode supports both unidirectional and bidirectional conversions, enabling a wide range of data transformation tasks. Here's how to utilize the conversion mode effectively:

1. **Define Conversion Mode**: In your `config.py` file, you'll define the `CONVERSION_MODE` setting, which governs how files are transformed. This setting offers two options: unidirectional and bidirectional conversion, denoted by `"->"` and `"&"` respectively.

   - Unidirectional Conversion: Denoted by `"->"`, this mode signifies a one-way transformation from one format to another. For instance, if you want to convert text files to HTML, you'd set `CONVERSION_MODE = "txt->html"`. This means the model will learn to convert text files specifically into HTML format, but not vice versa.

   - Bidirectional Conversion: Denoted by `"&"`, this mode implies a two-way transformation between formats. For example, setting `CONVERSION_MODE = "wav&mp3"` instructs the model to learn bidirectional conversion between WAV and MP3 audio formats. In this mode, the model learns to convert files from WAV to MP3 and vice versa, allowing flexibility in both directions of conversion.

2. **Prepare Your Data**: Ensure your data pairs are stored within the same directory path in both `TRAIN_FOLDERS` and `EVAL_FOLDERS`. Each pair should share identical paths, including filenames, differing only in their file extensions. For instance, if converting between WAV and MP3 formats, ensure files like "path/audio.wav" and "path/audio.mp3" are paired accordingly. This strict alignment guarantees the script correctly associates files for conversion based on the specified mode.

3. **Adjust Training Parameters**: Although the conversion mode operates under the same training principles as generative modelling, you might want to adjust certain parameters in `config.py` to optimize the conversion process. This could include tuning the `PATCH_SIZE` and `PATCH_LENGTH` settings to better accommodate the file sizes commonly encountered in your conversion tasks.

4. **Leverage Pre-trained Weights (Optional)**: Same as regular generative modelling, if you wish to fine-tune a pre-trained bGPT model, set `PRETRAINED_PATH` to the location of the pre-trained weights and ensure `LOAD_FROM_PRETRAINED=True`. To train a model from scratch, simply set `LOAD_FROM_PRETRAINED=False`.

5. **Start Training for Conversion**: When training bGPT in conversion mode, the model learns to map byte sequences from the source format to the target format (or vice versa in bidirectional mode). Execute `train-gen.py` to start the training process, ensuring that the `CONVERSION_MODE` is correctly set in your configuration file.

By leveraging the conversion mode, bGPT enables simulating and reverse engineering the behaviors of algorithms or hardware through paired inputs and outputs, opening up new possibilities for data processing and content generation tasks.

### Classification

Classification with bGPT leverages the model's ability to understand and differentiate between various types of data at a fundamental level. This involves extracting a global feature from the byte sequence, which is then processed by a classification head. Here's how to approach classification tasks with bGPT:

1. **Prepare Labelled Data**: Ensure your dataset consists of labelled data, which can be a mix of different formats. The model distinguishes between data types using the naming convention `label.ext`, where the label is derived from the filename, specifically `filename.split('_')[0]`. This means that the label for classification should be clearly reflected in the file name, such as "Business_1.txt". It is crucial to organize your files accordingly to facilitate accurate classification.

2. **Generative Modelling Before Classification (Strongly Recommended)**: Before embarking on classification tasks, it is highly recommended to perform generative modelling on the same dataset. Starting with weights trained through generative modelling provides a solid foundation for further fine-tuning in classification tasks. To do this, set `PRETRAINED_PATH` to your generative model weights and ensure `LOAD_FROM_PRETRAINED=True`. Directly training a classification model from scratch without this pre-training step has been observed to result in significantly poorer performance. When fine-tuning for classification, ensure that `WEIGHTS_PATH` and `LOGS_PATH` are set to different locations to prevent overwriting previous models. Note that the classification model will inherit the bGPT's patch-level decoder and discard the byte-level decoder, so it's essential to keep the model parameters unchanged during this phase.

3. **Start Training for Classification**: Run `train-cls.py` to begin the classification training process. The script will utilize the previously set configurations and apply them to your labelled dataset. The model will learn to classify the input data into the defined categories based on the labels extracted from the filenames.

### Inference

Inference with bGPT allows you to apply trained models to new data, performing tasks like data conversion or content generation based on the configurations set in `config.py`. Here's how to conduct inference using the provided settings:

1. **Set Up the Inference Configuration**: First, ensure your `config.py` includes the configuration for inference as shown above. Adjust the parameters according to the task you want to perform:

   - `INFERENCE_WEIGHTS_PATH`: This should point to the location of your trained model weights that you intend to use for inference. For example, `weights-conversion` indicates the model trained for converting files from one format to another.
   - `INPUT_EXT` and `TARGET_EXT`: These parameters define the extensions of the input and target files, respectively. In the given configuration, the model expects input files with the `.mid` extension and will aim to convert them into files with the `.abc` extension.
   - `MODE`: Determines the mode of inference. `convert` mode is used for converting files from one format to another, while `generate` mode is used for generating new content.
   - `NUM_SAMPLES`, `TOP_K`, `TOP_P`, and `TEMPERATURE`: These parameters control the sampling strategy for generation tasks, influencing the diversity and creativity of the output.

2. **Performing Conversion or Generation**: Depending on the `MODE` you've set, the inference process will either convert input files to a new format or generate new content:
   
   - **Conversion**: In `convert` mode, ensure your input files (e.g., `.mid`) are placed in a designated directory. The model will read these files, apply the conversion process, and output files in the target format (e.g., `.abc`) in the specified output directory.
   - **Generation**: In `generate` mode, the model will generate samples from scratch. The number of samples to generate is specified by `NUM_SAMPLES`. The generated samples will be placed in the output directory.

4. **Executing Inference**: To start the inference process, run the script `inference.py`. Make sure the script is configured to use the settings from `config.py` for conducting the desired inference task.

### CPU States Dataset

The CPU States Dataset is integral to exploring and modelling CPU behaviour, including a simplified yet detailed representation of CPU operations through sequences of machine instructions and subsequent register states. This dataset is crucial for training models like bGPT to predict how the CPU state updates with each instruction, demonstrating the model's proficiency in simulating digital processes within hardware environments.

The generation of the CPU States Dataset is facilitated by the `cpu-simulation.py` script, which allows for the generation, translation, and evaluation of CPU state sequences. The script operates in various modes to accommodate different stages of dataset preparation and model testing:

- **Generate Mode**: This mode is used to generate new instances of CPU states, simulating a wide range of program instructions to create a diverse dataset. It involves specifying the number of registers, memory size, and the range and quantity of instruction sequences to be generated. Example command:
  ```bash
  python cpu-simulation.py --mode generate --num_registers 10 --memory_size 1024 --dir_path cpu_states --num_min_program 1 --num_max_program 255 --num_train_instance 2100000 --num_test_instance 21000 --random_seed 0
  ```
  This generates training and testing instances with programs containing 1 to 255 instructions, tailored to the specified memory size and register count.

- **Translate Mode**: It is used to convert binary representations of CPU states into a human-readable format. This conversion is essential for manual inspection, debugging, and understanding the dataset's intricacies. Example command:
  ```bash
  python cpu-simulation.py --mode translate --states_path cpu_states/cpu.bin
  ```

- **Evaluate Mode**: To assess the accuracy of the model's predictions, the evaluate mode compares the predicted CPU states against the actual states within the dataset. Example command:
  ```bash
  python cpu-simulation.py --mode evaluate --dir_path cpu_states
  ```

The CPU States Dataset contains 2.1 million instances, each featuring a 1KB memory block and sequences of 16-byte CPU register states that include a variety of instruction types, such as data movement, logical, and arithmetic operations. This dataset simulates typical CPU behaviour, providing a rich resource for research and development in digital system simulation.

## BibTeX
```
@misc{wu2024language,
      title={Beyond Language Models: Byte Models are Digital World Simulators}, 
      author={Shangda Wu and Xu Tan and Zili Wang and Rui Wang and Xiaobing Li and Maosong Sun},
      year={2024},
      eprint={2402.19155},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
