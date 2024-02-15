# Persuasion Classifiers

Library to run the PT classifier, postprocess the results (aggregate, apply thresholds etc.) and visualize the results.

For container-based execution see [below](#runningc) 

## Getting started

Install the environment (using conda):

> `conda env create -f persuasion.yml`

See an example on `calling_example.ipynb`

Set the model path in the `model_fp = ...` variable on `defaults.cfg`

## Basic usage

### Load configuration and instantiate model

>```
> cfg = read_config('./defaults.cfg')
> model = instantiate_model(cfg)
>```

### Full Prediction (inference and postprocessing)

Get a JSON compatible dict:

>```
> output =  predict(model, text)
>
> # aggregate on sentences
> results_sentence = aggregate_results(text, output, level='sentence')
>
> # convert to json dict
> json_output = output_to_json(results_sentence, document_ids=guid, map_to_labels=True)
>```

### Raw Prediction (only inference)

Get a list of numpy arrays with raw scores per token:

>```
> output_raw = predict_raw(model, text)
>```

### Post-process Raw scores 

Get a JSON compatible dict from a list of numpy arrays:

>```
> output = postprocess_scores(output_raw, threshold=0.35)
>
> # aggregate on sentences
> results_sentence = aggregate_results(text, output, level='sentence')
>
> # convert to json dict
> json_output = output_to_json(results_sentence, document_ids=guid, map_to_labels=True)
>```

## The 'PersuationResults' class
Create a 'PersuationResults' holding the results of the prediction

>```
> results = PersuationResults(output, text, uids=guid)
>```

Slice the results 

>```
> results = results[6:40]
>```

Compute aggregations of the results

>```
> results = results.aggregate_results(levels=['paragraph', 'sentence', 'word'])
>```

### Return JSON compatible dict with the results

>```
> results.to_dict(level='sentence')
>```

Possible JSON output:

Results to dict `{guid -> {label -> [sentences]}}`

>```
> results.to_dict(level='sentence', orient='labels')
> ```

> ```
> {
>    'salzburger-f05f37d68053c917be4b96d8a59a9c31': {
>        <Persuation Techniques.Appeal_to_Authority: 0>: [0],
>        <Persuation Techniques.Appeal_to_Popularity: 3>: [1],
>        <Persuation Techniques.Exaggeration-Minimisation: 10>: [5],
>        <Persuation Techniques.Name_Calling-Labeling: 15>: [6, 7, 11]
>    }, 
>    'sn-at-1e41583d692396e05acfa5bd77fd8154': {
>        <Persuation Techniques.Appeal_to_Hypocrisy: 2>: [4],
>        <Persuation Techniques.Appeal_to_Time: 4>: [7, 12, 13],
>        <Persuation Techniques.Name_Calling-Labeling: 15>: [14, 15, 26]}
>    }
> }
>```

Results to dict `{guid -> {label -> [sentences]}}`


Get results to dict `{guid -> sentence -> [labels]}`

> ```
> results.to_dict(level='sentence', orient='segments')
> ``` 

>```
> {
>   'salzburger-f05f37d68053c917be4b96d8a59a9c31': {
>        0: [<Persuation Techniques.Appeal_to_Popularity: 3>],
>        1: [<Persuation Techniques.Appeal_to_Authority: 0>],
>        5: [<Persuation Techniques.Name_Calling-Labeling: 15>],
>        6: [<Persuation Techniques.Exaggeration-Minimisation: 10>],
>        7: [<Persuation Techniques.Name_Calling-Labeling: 15>],
>        11: [<Persuation Techniques.Appeal_to_Popularity: 3>]
>    },
>   'sn-at-1e41583d692396e05acfa5bd77fd8154': {
>        4: [<Persuation Techniques.Appeal_to_Time: 4>],
>        7: [<Persuation Techniques.Name_Calling-Labeling: 15>],
>        12: [<Persuation Techniques.Name_Calling-Labeling: 15>],
>        13: [<Persuation Techniques.Name_Calling-Labeling: 15>],
>        14: [<Persuation Techniques.Appeal_to_Hypocrisy: 2>],
>        15: [<Persuation Techniques.Appeal_to_Hypocrisy: 2>],
>        26: [<Persuation Techniques.Name_Calling-Labeling: 15>]
>    }
>}
>```

**The 'return_spans=True' functionality:**

Instead of returning the index of segment (word, sentence, paragraph) return its span in a (start, end) character offsets.

> ```
> results.to_dict(level='sentence', orient='labels',return_spans=True)
> ```

>```
> {'salzburger-f05f37d68053c917be4b96d8a59a9c31': {
>     <Persuation Techniques.Appeal_to_Authority: 0>: [(0,120)],
>     <Persuation Techniques.Appeal_to_Popularity: 3>: [(122, 213)],
>     <Persuation Techniques.Exaggeration-Minimisation: 10>: [(472, 530)],
>     <Persuation Techniques.Name_Calling-Labeling: 15>: [(533, 632),(634, 712),(1011, 1180)]
>     },
>  'sn-at-1e41583d692396e05acfa5bd77fd8154': {
>     <Persuation Techniques.Appeal_to_Hypocrisy: 2>: [(581,695)],
>     <Persuation Techniques.Appeal_to_Time: 4>: [(858, 901),(1238, 1287),(1289, 1378)],
>     <Persuation Techniques.Name_Calling-Labeling: 15>: [(1380, 1433),(1436, 1559),(2716, 2772)]
>     }
> }
>```


**The 'collapse_spans=True' functionality:**

Automatically collapse results to the maximum consecutive series of offsets.

(e.g. if segments [2,3,4] are annotated with the same label, the span is collapsed to (start_of_2, end_of_4) )

*See bolded text for comparison with the previous result'*

>```
>{'salzburger-f05f37d68053c917be4b96d8a59a9c31': {
>    <Persuation Techniques.Appeal_to_Authority: 0>: [(0,120)],
>    <Persuation Techniques.Appeal_to_Popularity: 3>: [(122, 213)],
>    <Persuation Techniques.Exaggeration-Minimisation: 10>: [(472, 530)],
>    <Persuation Techniques.Name_Calling-Labeling: 15>: [**(533, 712)**,(1011, 1180)]
>    },
>'sn-at-1e41583d692396e05acfa5bd77fd8154': {
>    <Persuation Techniques.Appeal_to_Hypocrisy: 2>: [(581,695)],
>    <Persuation Techniques.Appeal_to_Time: 4>: [(858, 901), (1238, 1378)],
>    <Persuation Techniques.Name_Calling-Labeling: 15>: [**(1380, 1559)**,(2716, 2772)]}
>    }
> ```

**get the segment spans by object attributes**
> ```
> results.words
> results.sentences
> results.paragraphs
> ```

## <a name="runningc"></a>Running in a container with NVIDIA GPU
The container needs the NVIDIA Container runtime installed and the container runtime in use (e.g. Docker) properly configured. The following assumes Docker is used as the container runtime: please refer to your CRI implementation for further information.

### Host install (Ubuntu 22 LTS)

From the nvidia [nvidia installation guide](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html).
 
 ```
 apt-get install linux-headers-$(uname -r)
 distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
 wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.0-1_all.deb
 dpkg -i cuda-keyring_1.0-1_all.deb 
 apt-get update
 apt-get -y install cuda-drivers
 apt clean
 apt autopurge
 reboot
 ```
 
 test it with:

 > ` nvidia-smi `
 
 Install the [nvidia-container-toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#install-guide).
 
 ```
 apt install nvidia-container-toolkit
 apt install docker-compose
 nvidia-ctk runtime configure --runtime=docker
 ```

 reboot and test it with:
```
 nvidia-ctk --version
 docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.6.2-base-ubuntu20.04 nvidia-smi
 ```

nvidia-smi should work as expected in the container.

### Container image build

From the root of the repository checkout build the container image with:

> `docker build -t persuasion:0.1.0 ./`

Launch the container with: 

> `docker container run --rm --gpus all --name test1 -v /srv/models/:/srv/models/ -p 8000:8000 -e "WEB_CONCURRENCY=1" persuasion:0.1.0`

Connect to http://localhost:8000/docs on the host to access the FastAPI documentation of the module.

Note that:
- --gpus param instructs which devices can be assigned.
- The container expects models to be available under /srv/models, esp. /srv/models/pt_model_multi_fine/ with the default config.
- Uvicorn is spawned on port 8000 inside the container
- The "WEB_CONCURRENCY" environment variable is use by uvicorn to decide the execution model