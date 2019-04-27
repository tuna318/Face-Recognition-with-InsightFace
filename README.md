# Face Recognition
Recognize and manipulate faces with Python and its support libraries.
The project uses [MTCNN](https://github.com/ipazc/mtcnn) for detecting faces, then applies a simple alignment and feeds those aligned faces into embeddings model that provided by [InsightFace](https://github.com/deepinsight/insightface). Finally, we put a softmax classifier on top of embedded vectors that will do classification task.

[Game Of Thones actors](https://github.com/anhtuhsp/FaceRecognition/blob/master/datasets/test/img_test.jpg)

## Getting started
### Requirements
- Python 3.3+
- Virtualenv
- python-pip
- mx-net
- tensorflow
- macOS or Linux
### Installing 
Check update:
```
sudo apt-get update
```
Install python:
```
sudo apt-get install python3.6
```
Install pip:
```
sudo apt install python3-pip
```
Most of the necessary libraries were installed and stored in `env/` folder, so what we need is installing `virtualenv` to use this enviroment.
Install virtualenv:
```
sudo pip3 install virtualenv virtualenvwrapper
```
And repeat
```
until finished
```
End with an example of getting some data out of the system or using it for a little demo
