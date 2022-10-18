
# Source Code for GLAM (ICDM 2022 Short)

Author Credit: Saurabh Sawlani

Prerequisites: torch, torch-geometric


Example commands to run a single model configuration,
```
	python main.py --data=PROTEINS --lr=0.1

	python main.py --data=DHFR --use_node_attr --ignore_node_labels

	python main.py --data=COLLAB --aggregation=Mean --lr=0.001
```

To run a number of configurations, to use for model selection later, set the required configurations in config.txt and run:
```
    python main.py --use_config
```

Once the pickle file is generated for outputs of all configurations, use model selection to select models. 

If you use "--aggergation=both", make sure that both files "GIN_MMD_{dataset}_ {seed}.pkl" and "GIN_Mean_{dataset}_ {seed}.pkl" have been generated in the output folder. Here are two examples:

```

    python model_selection.py --data=PROTEINS --seed=1213 --aggregation=MMD

    python model_selection.py --data=AIDS --aggregation=both

```

Run both files with --help to see all parameters that can be input.


Here is a chunk of code which should run all experiments needed to recreate results in GLAM:
```

	python main.py --data=PROTEINS --use_config --config_file=config/MMD_config.txt
	python main.py --data=PROTEINS --aggregation=Mean --use_config --config_file=config/Mean_config.txt
    python model_selection.py --data=PROTEINS

```
