Please download the following files from F. Minutoli's Drive if you're interested in running anything withing the Jupyter notebook. In any case, besides these ones to download (size varies from few kB to some GB), in order to run the notebook successfully, the following files have to be placed at its very same level, as well:

- build_vocab.py
- data_loaders.py
- models.py
- uitls.py

Files to download:

- preprocessing.zip (3 GB): Contains a content folder, with inside a data folder with COCO images, an annotation folder with COCO annotations (Karpathy's splits) and a vocab.pkl with the binary of our vocabulary.
	https://drive.google.com/file/d/1njpdzE1BHHrtC7CHt-WLe7V2w7e919wj/view?usp=sharing

- adaptive-25.pkl (250 MB): Contains the binary of our PyTorch model
	https://drive.google.com/file/d/1g0HfjOmJA4Eh2m88O2sElPaDUm2OJi-q/view?usp=sharing

Note: The following files only serve diplay purposes in the notebook.

- eval_scores.pkl: Contains validation SPICE scores from training
	https://drive.google.com/file/d/17Z9jpqp_B_TLzLa0MOQ8u4MqgcOROsMm/view?usp=sharing

- metrics_scores-25.pkl: Scores on all the major captioning metrics for our adaptive-25.pkl KWL model.
	https://drive.google.com/file/d/1CzkKbW-ZQM3cxkFCWLd3rE4U9rQD30J9/view?usp=sharing

- reank_vg_probs.pkl: Contains average visual grounding probabilities from training for every word in the vocabulary.
	https://drive.google.com/file/d/1PU7eSV_M7Z56PzFhX4aIitKNtu6TNS0b/view?usp=sharing
	
Note: The favourable behaviour would be to load all these files in the root of your Drive and then mount the Drive on Colaboratory and let the notebook do it all for you.