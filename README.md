# Face Recognition with InsightFace
Recognize and manipulate faces with Python and its support libraries.
The project uses [MTCNN](https://github.com/ipazc/mtcnn) for detecting faces, then applies a simple alignment and feeds those aligned faces into embeddings model that provided by [InsightFace](https://github.com/deepinsight/insightface). Finally, we put a softmax classifier on top of embedded vectors that will do classification task.

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
## Usage
First, go to directory that you have cloned, activate __env__ to use installed package, alternatively, you must install all packages that necessary for this project.
```
source env/bin/activate
```
Now move to __/src__ folder and try out first recognition with this command:
```
python3 recognizer_image.py 
```
The result should be something like below picture
__Show picture here__
You can also try with recognition in video with:
```
python3 recognizer_video.py
```
__Show gif here__
or streaming if your machine supports camera:
```
python3 recognizer_stream.py
```
Follow this link to see an example of video streaming [Streaming](#linkhere)
## Build your own faces recognition system
### 1. Get data
Our training datasets were built as following structure:
```
/datasets
  /train
    /person1
      + face_01.jpg
      + face_02.jpg
      + ...
    /person2
      + face_01.jpg
      + face_02.jpg
      + ...
    / ...
  /test
  /unlabeled_faces
  /videos_input
  /videos_output
```

### 2. Generate face embeddings
### 3. Train classifier with softmax
### 4. Run

## Others
### Using gpu for better performance
I use __CPU__ for all recognition tasks for __mxnet__ haven't supported for __cuda__ in Ubuntu 18.10 yet. But if your machine has an Nvidia GPU and earlier version of Ubuntu, you can try it out for better performance both in speed and accuracy.
In my case, I have changed __line 46__ in _face_model_ `ctx = mx.cpu(0)` to use cpu. 
### Thanks
- Many thanks to [Davis King](https://github.com/davisking) for creating dlib with lots of helpful function in face deteting, tracking and recognizing
- Thanks to everyone who works on all the awesome Python data science libraries like numpy, scipy, scikit-image, pillow, etc, etc that makes this kind of stuff so easy and fun in Python.
- Thanks to Jia Guo and [Jiankang Deng](https://jiankangdeng.github.io/) for their InsightFace project
- Thanks to [Adrian Rosebrock](https://www.pyimagesearch.com/author/adrian/) for his useful tutorials in [pyimagesearch](https://www.pyimagesearch.com/) that help me a lots in building this project.
