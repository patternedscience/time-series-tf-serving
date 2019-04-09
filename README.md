# Seq2seq Model on Time-series Data: Training and Serving with TensorFlow

### Introduction

Seq2seq models are a class of Deep Learning models that have provided state-of-the-art solutions to language problems recently. They also perform very well on numerical, time-series data which is of particular interest in finance and IoT, among others. This repository contains two notebooks:

- Model development and optimization with TensorFlow (its low-level API);

- Serving the model with TensorFlow Serving and showing how to write a client to communicate with TF Serving over the network and use/plot the received predictions.

There is (or will be) a video of a presentation explaining the above code; check out the updates at [PatternedScience's LinkedIn page](https://www.linkedin.com/company/patterned-science/) to find the video.

### Conda Environments

- `requirements_training_node.txt`: env where the model is trained and exported;

- `requirements_client_node.txt` : env where the client runs and communicates with TF Serving (this env contains TensorFlow, but it does not need to have it, i.e., this is not the minimum-env that's required).

### Node hardware specs

- CPU node/worker (client) : 8 vCPUs, ~60GB RAM

- GPU node/worker (training): V100 GPU, 8 vCPUs, ~60GB RAM

This is where the above notebooks were run. The small mock dataset and the sine function that is learned in the notebooks do not, in fact, need GPU acceleration and need much less RAM.

### License

If you use all or part of the code in this repository, we suggest that you include the following notice with your document, code or product:

> This code/product is partly or fully based on the code which was originally run on the UniAnalytica platform (https://www.unianalytica.com) and is published by PatternedScience Inc. at https://github.com/patternedscience/time-series-tf-serving and licensed under the terms of Apache License 2.0; a copy of the license is available in the GitHub repository.

Feel free to adapt the "*This code/product is partly or fully based on*" part to your situation and use. If you need some modifications to the above text/license to accommodate better your use, please contact [PatternedScience Inc.](https://www.patterned.science/)

Copyright © 2019 PatternedScience Inc.
