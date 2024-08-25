# Elissabeth Projects

Each folder here contains experiments where Elissabeth models train on given data.
For each project, several files are given:

- `data.py`, containing methods or classes for data handling and creation
- `main.py`, the script callable from a terminal, which starts training
- `config.json`, the configuration file for the current Elissabeth model
- `watcher.ipynb`, a python notebook containing tools to analyze a trained Elissabeth model

## Starting an experiment

We advise to browse through the code to get used to our pipeline. Starting an experiment is
simple. Below is a walkthrough of a typical test run.

1. Configure the model by modifying the `config.json` file.
2. Configure data and training hyperparameters, which are found at the top of each `main.py` file
   in a Python dictionary called `config`.
3. Start an experiment run by calling

        $ python main.py train
4. Test the trained model by calling

        $ python main.py test --load [your-model-id]
    where `your-model-id` is depending on your experiment tracker and logger. An experiment is
    typically saved in your current experiment folder. The model id is the name of the folder
    containing the checkpoint file.
5. Observe model parameters and model behaviour in specific test cases in the `watcher.ipynb`
   notebook, again loading the model with the given model id.
