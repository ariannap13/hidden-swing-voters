# Persuasion Classifiers

Library to run the PT classifier.

Requirements are specified in *requirements.txt*.

The main script is `propaganda_detection.py`. It requires the specification of the configuration file in `defaults.cfg` (in which the only element to be changed is the trained model directory *model_fp*) and the location of the json file of tweets to be annotated (*tweets_vips.json*). It then produces an annotated json file of tweets as *tweets_vips_annotated.json*.

The folder *./i3/* contains some useful processing scripts.
