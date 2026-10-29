# Running the Samson notebook

1. Put the original `Samson.mat` and `Samson_GT.mat` files in `data/raw/`. The loader expects a 95×95×156 cube (or `V` shaped 156×9025) and GT `A` shaped 3×9025 plus `M` shaped 156×3. The GT file is accessed only after training, in evaluation.
2. Prepare a Python environment with PyTorch, NumPy, SciPy, scikit-learn, Matplotlib, and Jupyter. `requirements.txt` lists the non-PyTorch dependencies; choose/install PyTorch for the machine and driver separately. The project never installs or changes PyTorch, CUDA, or drivers.
3. Open `code/EGU_Net_Samson.ipynb` in Jupyter and run the cells in order. It checks a real CUDA operation before selecting GPU 0. If that fails, the notebook stops before full training and prints a driver/build diagnostic. A CPU smoke test is available with `python code/test_smoke.py` once dependencies are installed.

Training settings are centralized in `code/settings.py`. `BATCH_SIZE=128` is a memory-conscious alternative; lower it to reduce activation memory. Mixed precision is off by default. The target is 200 epochs, and rerunning the training cell resumes `outputs/samson/checkpoints/latest.pt`. The notebook skips additional training after reaching the configured epoch count and regenerates evaluation artifacts when run.

Checkpoints are saved after each epoch and every configured number of optimizer steps. Recovery saves the batch order and cursor as well as model, optimizer, schedule, history, configuration, and RNG state. Resume rejects incompatible model settings. An interruption during a CUDA kernel cannot be checkpointed until the current optimizer step finishes.

Figures are written to `outputs/samson/figures/`, metrics to `outputs/samson/metrics/`, predictions/reconstruction/recovered spectra and selected activations to `outputs/samson/features/`, and preprocessing cache to `data/processed/`.
