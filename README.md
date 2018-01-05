# yolo-detection-api 

## Description
Runs a Flask API using YOLO to detect objects in a frame. Use Docker to deploy where possible.

## Quick Start with Docker
1. `sudo docker-compose up --build -d` or for GPU `sudo docker-compose -f gpu-compose.yml up --build -d`
2. Navigate to http://localhost:5001

## Installation
1. Download the model folder from [here](https://drive.google.com/open?id=1NYtW4w2EjasFzvNQt_J6jduWeNWUIxyQ) and put it the current directory (where this file lives).
2. Create a virtual environment with Python 3.6
3. Run `pip install -r requirements.txt`
4. Install [darkflow](https://github.com/thtrieu/darkflow)

## Setup
1. Run `redis-server` or set the `REDIS_URL` environmental variable with the Redis connection string.
1. (Optional) Set the `THROTTLE_SECONDS` environment variable to limit request calls to this server. The default will be 5 if not set. Use an integer.
1. Run `python api.py` 
1. Navigate to http://localhost:5001
 

## Requirements
1. [darkflow](https://github.com/thtrieu/darkflow)
1. OpenCV 3
1. Python 3.6
1. Tensorflow
1. Redis