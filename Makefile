download:
	# Download results
	rsync -avz csf:"~/GECCO-extension/results/*" ./results/

upload:
	rsync -avz "input/malaria/data_original/subgraph_1000_seed0.pkl" csf:"~/GECCO-extension/input/malaria/data_original/"

