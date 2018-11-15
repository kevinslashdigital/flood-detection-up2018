# README #

Explaination with mathamatic. https://drive.google.com/file/d/1uD74YaAkmbqgFE_hn7GpBt_GM1S8zAJ8/view?usp=sharing

## What is this repository for? ##

This repo aims to share some knowledge about machine learning based on Naive Bayes Algorithm.
version 1.0

## How do I get set up? ##

* install python
* pip3 install khmerml
* pip3 install flask
#### unicode support ####
* for unicode support, you need to install some dependencies, here the guideline : http://niptict.edu.kh/khmer-word-segmentation-tool/
* you need to modify the path in Libs/Unicode/KhmerSegment.py
```
def __init__(self):
	self.PATH = '/Data/km-5tag-seg-1.0' # here the path of khmer segmentation lib that i download from niptict
```
### For frontend only
This is for frontend development.
##### Install additional prerequisites
* [Node](https://nodejs.org/en/) please refer to their [installation guide](https://nodejs.org/en/download/package-manager/)
* [Yarn](https://yarnpkg.com)  please refer to their [installation guide](https://yarnpkg.com/en/docs/install).

```sh
$ git clone https://bitbucket.org/ventureslash/kyc-api-fork.git
$ cd kyc-api-fork
$ yarn global add webpack
$ yarn global add webpack-cli
$ yarn global add webpack-dev-server
$ yarn install # install the require dependancies
```
#### How to run frontend
```sh
$ yarn run dev # to start development 
# to run with an Endpoint API
#ENV_API=API_URL yarn run dev #ie. 
$ENV_API=localhost:3000 yarn run dev
$ yarn run build # to build for production
```

## Who do I talk to? ##

if you have any question, you can send to these emails.
- email1@gmail.com
- email2@gmail.com
- email3@gmail.com  

## references ##
https://www.youtube.com/watch?v=UkzhouEk6uY&index=3&list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGL
https://github.com/spmallick/learnopencv/blob/master/