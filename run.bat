@echo off
:: Check if firstLaunch.txt exists
if exist firstLaunch.txt (
    echo First launch detected. Setting up environment...

    :: Set up a virtual environment and activate it
    python -m venv venv
    call venv\Scripts\activate

    :: Clone the repository without checking out files immediately
    git clone --no-checkout https://huggingface.co/adamo1139/Meta_Spirit-LM-ungated

    :: Enter the cloned repository
    cd Meta_Spirit-LM-ungated

    :: Enable sparse checkout with cone mode
    git sparse-checkout init --cone

    :: Only include the necessary paths (everything except the unwanted folder)
    git sparse-checkout set --no-cone "/*" "!spiritlm_model/spirit-lm-expressive-7b"

    :: Checkout the files
    git checkout

    cd ..

    git clone https://github.com/facebookresearch/spiritlm

    :: Install the SpiritLM repository dependencies and package in editable mode
    pip install -r spiritlm/requirements.txt
    pip install -e spiritlm

    :: Uninstall the default version of torch
    pip uninstall -y torch

    :: Install the remaining dependencies from requirements.txt
    pip install -r requirements.txt

    :: Remove the firstLaunch.txt file to prevent future installations
    del firstLaunch.txt

    echo Setup complete.
) else (
    echo Not the first launch. Skipping installation.
)

:: Activate the virtual environment
call venv\Scripts\activate

:: Run the main application
echo Running Gradio.
python main.py
