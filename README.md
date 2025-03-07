# PII-Anonymiser
Framework Designed to Anonymise personally identifiable information from video data obtained from vehicles.

## Key Features
* 5  different Methods for blurring the information.
* 2 Different Types of Blur (Blackbox and Gaussian)
* Ability to adjust the size of each segment for flexibility.
* Pose Detection using Mediapipe

## How to use 
To clone and run this application, you'll need [Git](https://git-scm.com), Along with that ensure that all dependencies mentioned in the requirements.txt are installed as well.
The video location is given in as the input.Output will be saved as final_output.mp4

---
```bash
$ git clone <repo-addr>
$ cd .\PII-Anonymiser\
$ python3 Anonymize.py --help
```
---

## Usage
---
```
usage: Anonymize.py [-h] [--blur {basic_blur,cascade_blur,ocr_blur,custom_blur,final_blur}] [--blackboxblur] [--split  Range [2-120]] [--ocr] [--mediapipe] input

GDPR based Framework to annonmize Personally indentifiable information

positional arguments:
  input                 The input file which should be processed

options:
  -h, --help            show this help message and exit
  --blur {basic_blur,cascade_blur,ocr_blur,custom_blur,final_blur}
                        Specifies which type of method is used to blur out the number plates of vehicles.

                        basic_blur : Blurs out approximately half of the vehicle.Fast but a lot of information is lost.
                        cascade_blur: Uses the cv2 cascade classifier.Works on the haarcascade_russian_plate_number dataset.This method is not effective enough to be used in this project.
                        ocr_blur : Uses optical character recognition to blur out all letters from the car.Slow and loses a lot of information in the process.
                        custom_blur : Uses YOLO model which is fined tuned with a number plate dataset.Best model fast and accurate.
                        final_blur : Uses custom_blur model along with basic_blur to cover most of the cases.This method is best method for blurring.
  --blackboxblur        Enables Black Box Blurring instead of Gaussian Blur
  --split  Range [2-120]
                        Specifies the size of the segments the video is split into.The default value is 60 seconds and this is recommended unless the video is shorter in length.Range is [2-120] seconds
  --ocr                 Enables detection and bluring of political messages
  --mediapipe           Enables mediapipe and shows pose tracking
```
---
