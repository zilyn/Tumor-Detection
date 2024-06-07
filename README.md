# Tumor Detection Model

# Execution Instructions

# Python version 3.8

To create a virtual environment and install requirements in Python 3.8 on different operating systems, follow the instructions below:

### For Windows:

Open the Command Prompt by pressing Win + R, typing "cmd", and pressing Enter.

Change the directory to the desired location for your project:


cd C:\path\to\project

Create a new virtual environment using the venv module:


python -m venv myenv

Activate the virtual environment:
myenv\Scripts\activate


Install the project requirements using pip:
pip install -r requirements.txt

### For Linux/Mac:
Open a terminal.

Change the directory to the desired location for your project:

cd /path/to/project

Create a new virtual environment using the venv module:

python3.8 -m venv myenv


Activate the virtual environment:
source myenv/bin/activate

Install the project requirements using pip:
pip install -r requirements.txt

These instructions assume you have Python 3.8 installed and added to your system's PATH variable.

## Execution Instructions if Multiple Python Versions Installed

If you have multiple Python versions installed on your system, you can use the Python Launcher to create a virtual environment with Python 3.8. Specify the version using the -p or --python flag. Follow the instructions below:

For Windows:
Open the Command Prompt by pressing Win + R, typing "cmd", and pressing Enter.

Change the directory to the desired location for your project:

cd C:\path\to\project

Create a new virtual environment using the Python Launcher:

py -3.8 -m venv myenv

Note: Replace myenv with your desired virtual environment name.

Activate the virtual environment:


myenv\Scripts\activate


Install the project requirements using pip:

pip install -r requirements.txt


### For Linux/Mac:
Open a terminal.

Change the directory to the desired location for your project:

cd /path/to/project

Create a new virtual environment using the Python Launcher:


python3.8 -m venv myenv


Note: Replace myenv with your desired virtual environment name.

Activate the virtual environment:

source myenv/bin/activate


Install the project requirements using pip:

pip install -r requirements.txt


By specifying the version using py -3.8 or python3.8, you can ensure that the virtual environment is created using Python 3.8 specifically, even if you have other Python versions installed.


### CVC-ClinicDB Dataset
The dataset consists of frames extracted from colonoscopy videos. It consists of several examples of polyp frames & corresponding ground truth for them. The Ground Truth consists of a mask corresponding to the region covered by the polyp i the image.

Download the data and place the PNG folder inside the input folder. The file structure is the following:


```
input
└──PNG
   └──Original
   |  └──1.png, 2.png,...
   └──Ground Truth
      └──1.png, 2.png,...
```
### Train the model

To train the model, run:
```
python train.py
```

### Use the model to make predictions
To make predictions on an image, run:
```
python predict.py --test_img path_to_test_img
```
