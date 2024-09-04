# RobotVoice

_Verified Set of TTS Voices for Common Robots_

## Install server

### With Docker
To run locally, install Docker Desktop and run in the folder `RobotVoice`:

```sh
git clone https://github.com/polvanrijn/RobotVoice
cd RobotVoice
docker-compose up -d --build
```

### Without Docker

If you don't want to use Docker, you can have to install the following pre-requisites: espeak unzip git wget

When these packages are installed, you will have to configure a python environment (conda or other) which uses **python 3.10**.

When the environment is activated, you can install the project using:

```sh
git clone https://github.com/polvanrijn/RobotVoice
cd RobotVoice
sh setup.sh
```

Finally you can start the server using:

```sh
# Set the number of workers based on the number of CPUs, set max-requests = 1 to avoid memory leak
# The port is set to 5000 but can be changed as needed by modifying the value of "-b"
gunicorn server:app --worker-tmp-dir /dev/shm --workers=1 -b :5000 -t 600 --max-requests 1
```

## Access the synthesis server

You should now be able to test the server locally on http://localhost:5000/

## Scraping for robots

Setup a virtual environment and install the dependencies:
```sh
pip install opencv-python matplotlib beautifulsoup4 requests
```

To create a json with all meta information from the robots listed at https://robots.ieee.org/robots/, run:
```
python scrape_ieee.py
```

To select the images for the robots, run the following script.
```sh
mkdir "images"
python select_robots.py
```

An array with images will pop up. Either specify the image index (starts at 1!) or reject the image by entering `n`.
