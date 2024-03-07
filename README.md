# CounterfacTS
Tool for Probing the Robustness of Deep Learning Time-series Forecasting Models

This is the code from our paper [Kjærnli et al. 2024: Probing the Robustness of Time-series Forecasting Models with CounterfacTS](https://arxiv.org/abs/2403.03508)




## Running the application

To run the application start by installing and activating the environment after cloning the repository in your machine:

```shell
conda env create -f env.yaml
conda activate counterfacts
```

We can then use the following command to run the application

```shell
bokeh serve src/ --args <config-path>
```

where the `config_path` is the path to a config.yaml file in the experiments folder.
As a concrete example, this command will run the application using a simple dense network on the electricity dataset:
```shell
bokeh serve src/ --args experiments/electricity_nips/feedforward/config.yaml
```
Finally, copy the http address that appears in the terminal and paste it in the browser to open the interactive GUI.


