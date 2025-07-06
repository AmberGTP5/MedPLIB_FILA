# Uni-MedVLM: An Integrated Biomedical MLLM with High-Resolution Perception, Multi-Task Execution, and Explainable Reasoning

<p align="center">
  <img src="https://img.shields.io/badge/status-in%20progress-yellow" alt="Status"/>
  <img src="https://img.shields.io/badge/Stage-3%20of%205-blue" alt="Stage"/>
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/DeepSpeed-integrated-blueviolet" alt="DeepSpeed"/>
</p>

This repository contains the development of **Uni-MedVLM**, a novel and ambitious end-to-end framework for biomedical Multimodal Large Language Models (MLLMs). This project aims to systematically address the core challenges in the field by integrating three state-of-the-art methodologies into a "trinity" solution:

1.  👀 **Perception (How to See Clearly?)**: Adopts the **Hybrid Encoder** from the FILA architecture to process high-resolution medical images, capturing fine-grained details often missed by standard models.
2.  🖐️ **Execution (How to Excel at Tasks?)**: Leverages the **Mixture-of-Experts (MoE)** architecture inspired by MedPLIB, enabling the model to master diverse tasks like Visual Question Answering (VQA) and pixel-level Grounding through expert specialization.
3.  🧠 **Reasoning (How to Think Reliably?)**: Integrates the **GRPO Reinforcement Learning** framework inspired by MedVLM-R1, incentivizing the model to generate explainable and trustworthy reasoning chains beyond simple answers.

##  Roadmap & Current Status

This project is developed through a five-stage "incubation plan".

-   ✅ **Stage 1: High-Resolution Vision Alignment**: Completed.
-   ✅ **Stage 2: Supervised Fine-Tuning (SFT) for VQA Expert**: Completed.
-   ➡️ **Stage 3: Supervised Fine-Tuning (SFT) for Grounding Expert**: **In Progress**.
-   ⬜️ **Stage 4: MoE Router Training**: Pending.
-   ⬜️ **Stage 5: Reinforcement Learning for Reasoning**: Pending.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [Your-Repo-Link]
    cd Uni-MedVLM
    ```
2.  **Create Conda Environment:**
    ```bash
    conda create -n unimed python=3.10 -y
    conda activate unimed
    ```
3.  **Install Dependencies:**
    The dependencies are largely based on the MedPLIB project.
    ```bash
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    # Install requirements from the copied MedPLIB project
    pip install -r ./scripts/Stage3/requirements.txt 
    # Install additional packages for flash attention
    pip install ninja==1.11.1.1
    pip install flash-attn==2.5.2 --no-build-isolation
    ```

## 🗃️ Dataset Preparation

This project utilizes two main data sources, as specified in the MedPLIB project.

1.  **Annotations (`MeCoVQA`)**: Download the `MeCoVQA` dataset from the original project's [Google Drive](https://drive.google.com/file/d/1zIZJ5OBmV3OPc41H_Iaz9mdEh7wHmHqv/view?usp=drive_link). This will contain the JSON files for different tasks (`MeCoVQA-Complex.json`, `MeCoVQA-Region.json`, `MeCoVQA-Grounding.json`).

2.  **Images & Masks (`SA-Med2D`)**: Download the `SA-Med2D-16M` image and mask repository from [Hugging Face](https://huggingface.co/datasets/OpenGVLab/SA-Med2D-20M). Decompress the split archive to get the `images/` and `masks/` folders.

3.  **Create Clean Subsets**: For efficient development, we use scripts to create smaller, verified subsets of the data. For example, to prepare the data for Stage 3:
    * Configure the paths in `scripts/Stage3/create_grounding_subset.py`.
    * Run the script: `python scripts/Stage3/create_grounding_subset.py`.
    * This will generate a clean JSON annotation file and corresponding image/mask sub-folders inside the specified destination directory.

## 📀 Training Pipeline

All training stages should be launched from their respective `scripts/` directory (e.g., `cd scripts/Stage3/scripts`).

### ✅ Stage 1: High-Resolution Vision Alignment

* **Goal**: To train the `mm_projector` and `CVFM` modules, aligning the new `HybridEncoder` with the LLM.
* **Launch Script**: `scripts/Stage1/scripts/train_stage1_alignment.sh`
* **Execution**:
    ```bash
    cd scripts/Stage1/scripts
    bash train_stage1_alignment.sh
    ```
* **Output**: A `stage1_projector_cvfm.bin` file containing the trained weights, located in the `runs/` directory. This artifact is crucial for all subsequent stages.

### ✅ Stage 2: VQA Expert SFT

* **Goal**: To fine-tune a lightweight LoRA adapter on the Stage 1 model, creating a specialized VQA expert.
* **Launch Script**: `scripts/Stage2/scripts/train_stage2_vqa_sft.sh`
* **Execution**:
    ```bash
    cd scripts/Stage2/scripts
    bash train_stage2_vqa_sft.sh
    ```
* **Output**: A LoRA adapter directory (containing `adapter_model.safetensors` and `adapter_config.json`) saved in the `runs/` directory. This represents the "VQA Expert".

### ➡️ Stage 3: Grounding Expert SFT (In Progress)

* **Goal**: To fine-tune a new set of modules (`mask_decoder` and `text_hidden_fcs`) on the Stage 1 model, creating a "Grounding Expert" capable of pixel-level segmentation based on text instructions.
* **Development Steps**: This stage follows a "scalpel-like" development process:
    1.  `✅` **Prepare Environment**: All code dependencies have been localized into the `scripts/Stage3` directory.
    2.  `✅` **Verify Dataset Script**: `grounding_dataset.py` has been verified with `test_dataset.py`.
    3.  `✅` **Verify Model Init**: The `UniMedVLMForGrounding` model framework and weight loading have been verified with `test_model_init.py`.
    4.  `➡️` **Verify Single Training Step**: This is the current step. We are running a one-step simulation to ensure the end-to-end training pipeline is functional.
* **Simulation Script**: `scripts/Stage3/scripts/train_stage3_simulation.sh`
* **Execution**:
    ```bash
    cd scripts/Stage3/scripts
    bash train_stage3_simulation.sh
    ```
* **Expected Output**: The script should complete after one step and print a valid loss value, confirming the pipeline is ready for full training.

## 👍 Acknowledgements
This project is built upon the foundational work of several incredible open-source projects. Our deepest gratitude goes to the authors and contributors of **MedPLIB**, **FILA**, **LISA**, and **SAM-Med2D**. Our work aims to integrate and extend their powerful ideas.
