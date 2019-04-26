
# Sample client code to communicate with TF Serving on the GPU worker
## Time-series prediction using a seq2seq model

### Version: v1.2
#### model_name=seq2seq_timeseries. The input and output tensors in this version are multidimensional.

### Licenses/Credits
* Copyright (c) 2019, PatternedScience Inc. This code was originally run on the [UniAnalytica platform](https://www.unianalytica.com/), is published by PatternedScience Inc. on GitHub and is licensed under the terms of Apache License 2.0; a copy of the license is available in the GitHub repository;

* The visualization code snippet are based on [the work by Guillaume Chevalier](https://github.com/guillaume-chevalier/seq2seq-signal-prediction) (MIT License).

### Run this on the command line on the GPU worker to launch TF Serving

If something else is using the GPU on the worker (e.g., the training code/notebook), it's better to shut it down first.
docker run --runtime=nvidia -p 8501:8501 --mount \
type=bind,\
source=/workspace/notebooks/unianalytica/group/time-series-tf-serving-folder/time-series-tf-serving/exported_models/exp1,\
target=/models/seq2seq_timeseries \
-e MODEL_NAME=seq2seq_timeseries -t $USER/tensorflow-serving-gpu
### Package imports, parameters, var initializations


```python
import json
from urllib.request import urlopen
import time
from datetime import datetime
import pytz
import math
import random
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

model_server_host_address = '10.0.0.163'
rest_api_port = 8501
model_name = 'seq2seq_timeseries'
model_ver = 1

nb_predictions = 5

seq_length_in = 10
seq_length_out = 5
input_dim = 1
output_dim = 1
dataset_x = []
dataset_y = []
```

### Creating a dataset to send for predictions


```python
for _ in range(nb_predictions):
    rand = random.random() * 2 * math.pi

    sig1 = np.sin(np.linspace(0.0 * math.pi + rand,
                              3.0 * math.pi + rand, seq_length_in + seq_length_out))
    sig2 = np.cos(np.linspace(0.0 * math.pi + rand,
                              3.0 * math.pi + rand, seq_length_in + seq_length_out))
        
    x1 = sig1[:seq_length_in]
    y1 = sig1[seq_length_in:]
    
    if input_dim == 1:
        x_=np.tile(x1[:,None], input_dim)
    
    elif input_dim == 2:
        x2 = sig2[:seq_length_in]
        x_ = np.array([x1, x2])
        x_= x_.T
    
    if output_dim == 1:
        y_=np.tile(y1[:,None], output_dim)
    
    elif output_dim == 2:
        y2 = sig2[seq_length_in:]
        y_ = np.array([y1, y2])
        y_= y_.T

    dataset_x.append(x_)
    dataset_y.append(y_)

X = np.array(dataset_x).transpose((1, 0, 2))
Y = np.array(dataset_y).transpose((1, 0, 2))
```

### Sending the data to the server and getting the results back in `outputs`


```python
now = datetime.now(pytz.timezone('US/Eastern'))
seconds_since_epoch_start = time.mktime(now.timetuple())
now_microsecond_start = now.microsecond

def CallREST(name, url, req):
  print('Sending {} request to {} with data:\n{}'.format(name, url, req))
  resp = urlopen(url, data=json.dumps(req).encode())
  #.dumps() changes JSON to string and .encode() returns a byte object
  resp_data = resp.read()
  print('Received response:\n{}'.format(resp_data))
  resp.close()
  return resp_data
  
# Test Predict implementation over REST API:
# Prepare request
url = 'http://{}:{}/v1/models/{}/versions/{}:predict'.format(
    model_server_host_address, rest_api_port, model_name, model_ver)
json_req = {'instances': X.tolist()}
    
# Send request
resp_data = None
try:
  resp_data = CallREST('Predict', url, json_req)
except Exception as e:
  print('Request failed with error: {}'.format(e))

outputs = np.array(json.loads(resp_data)['predictions'])
now = datetime.now(pytz.timezone('US/Eastern'))
```

    Sending Predict request to http://10.0.0.163:8501/v1/models/seq2seq_timeseries/versions/1:predict with data:
    {'instances': [[[-0.612751131181866], [-0.7214921258749699], [0.8580921038292475], [-0.9956912407412133], [-0.49588779130252736]], [[0.013660859981261838], [-0.9958037129400517], [0.3507240241631004], [-0.7206462126080599], [-0.9291308802670913]], [[0.6341121120037422], [-0.8356092603952078], [-0.3096779363320677], [-0.1311565527354478], [-0.9569597557495637]], [[0.9778767651763765], [-0.3108075406975469], [-0.8349559442633817], [0.515561568486957], [-0.5672316487327589]], [[0.8949575699740332], [0.34961101978359654], [-0.9959117510657991], [0.9373210835228484], [0.07000063408652842]], [[0.42153524218119276], [0.8574813444666849], [-0.7223143772228287], [0.9500926960714597], [0.6766890477759041]], [[-0.2358185233599519], [0.9912008016825387], [-0.1335444896383926], [0.5483036785803462], [0.9881129686985007]], [[-0.7902759336050559], [0.6924226399392328], [0.5134958046039868], [-0.09273054033716967], [0.8683866065509659]], [[-0.999906686098544], [0.09151483647369571], [0.9364788619476901], [-0.693302990224085], [0.36975100721174453]], [[-0.7732411198391835], [-0.5493242794031351], [0.9508415092690853], [-0.9913616689556611], [-0.29022065032615246]]]}
    Received response:
    b'{\n    "predictions": [[[-0.185426], [-0.974384], [0.533045], [-0.803222], [-0.849805]], [[0.482786], [-0.911632], [-0.0802037], [-0.283707], [-0.981176]], [[0.94824], [-0.494898], [-0.66986], [0.346008], [-0.715854]], [[1.01962], [0.137829], [-0.96413], [0.870894], [-0.147267]], [[0.674137], [0.748701], [-0.833301], [1.04135], [0.515905]]\n    ]\n}'


### Measuring the communication/prediction delay and plotting the received predictions:


```python
seconds_since_epoch_end = time.mktime(now.timetuple())
now_microsecond_end = now.microsecond
if seconds_since_epoch_end != seconds_since_epoch_start:
    print('Communication and prediction took {} second(s).'.format(seconds_since_epoch_end - seconds_since_epoch_start))
else:
    duration_microseconds_part = now_microsecond_end - now_microsecond_start
    print('Communication and prediction took {} milliseconds.'.format(duration_microseconds_part/1000))

for j in range(nb_predictions): 
    plt.figure(figsize=(12, 3))

    for k in range(input_dim):
        past = X[:,j,k]
        label1 = "Seen (past) values" if k==0 else "_nolegend_"
        plt.plot(range(len(past)), past, "o--b", label=label1)

    for k in range(output_dim):
        expected = Y[:,j,k]
        pred = outputs[:,j,k]

        label2 = "True future values" if k==0 else "_nolegend_"
        label3 = "Predictions" if k==0 else "_nolegend_"
        plt.plot(range(len(past), len(expected)+len(past)), expected, "x--b", label=label2)
        plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y", label=label3)

    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()
```

    Communication and prediction took 4.874 milliseconds.



![png](tf-serving-client_files/tf-serving-client_10_1.png)



![png](tf-serving-client_files/tf-serving-client_10_2.png)



![png](tf-serving-client_files/tf-serving-client_10_3.png)



![png](tf-serving-client_files/tf-serving-client_10_4.png)



![png](tf-serving-client_files/tf-serving-client_10_5.png)

