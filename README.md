# RobotVoice
_Verified Set of TTS Voices for Common Robots_

## Run synthesis server
To run locally, install Docker Desktop and run in the folder `RobotVoice`:
```shell
docker-compose up -d --build
```

You should now be able to test it: http://localhost:5000/

## Run synthesis server on a server
Provision at least a m5.xlarge instance on AWS. Install Docker and Docker Compose. Then run:
```shell
git clone https://github.com/polvanrijn/RobotVoice
cd RobotVoice
docker-compose up -d --build
```

If you don't want to use Docker, you can also install the dependencies manually:
```shell
git clone https://github.com/polvanrijn/RobotVoice
cd RobotVoice
sh setup.sh
gunicorn server:app --worker-tmp-dir /dev/shm --workers=6 -b :5000 -t 600 --max-requests 30 # serve on port 5000
```

## Scraping for robots
Setup a virtual environment and install the dependencies:
```shell
pip install opencv-python matplotlib beautifulsoup4 requests
```

To create a json with all meta information from the robots listed at https://robots.ieee.org/robots/, run:
```
python scrape_ieee.py 
```

To select the images for the robots, run the following script. 
```shell
mkdir "images"
python select_robots.py 
```

An array with images will pop up. Either specify the image index (starts at 1!) or reject the image by entering `n`.

