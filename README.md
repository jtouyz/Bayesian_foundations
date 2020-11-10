# Course 1: Bayesian Foundations Notebooks
This repo is for UpSkill ML's 💻 Course 1 of Bayesian Foundations 💻.

You have two options to run this code either:
1. Working through a Binder online or
2. Downloading the code and working with it locally.


# Working the code through "My Binder"
My Binder is a free service that will set up a light-weight virtual dockerized python container. It will contain the set of notebooks for this course for you to work with so you don't need to download the code locally.

To access it click the link below and it will create a `jupyter` notebook environment for you that is already pre-configured with the appropriate libraries.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jtouyz/Bayesian_foundations/HEAD)

It is free to work with and requires no sign-up on your end. It will take a couple of minutes to spin up the container so be patient.

Note that sessions do expire if you are not actively work with the notebooks so you may need to "reload" a session if you step away from it for more than 30 minutes.

# Working with Code Locally 

## Pulling code
In this section we will go through pulling down the code for use in the UpSkill ML Bayesian Foundations course.

### Using a terminal: 
1. Open a terminal
2. Navigate to a folder where you would like to clone the the code
3. Run the following command

```
git clone https://github.com/jtouyz/Bayesian_foundations
```

4. A new folder will be created with the code

### Downloading the ZIP
Navigate to https://github.com/jtouyz/Bayesian_foundations and press the green `code` button. Select `Download Zip`. Unzip it on a place on your desktop.


## Setting up your virtual environment
It is recommended that you set up a virtual environment or run the code in a container. There are several approaches to creating a virtual environment and is generally considered best practice.

Below we demonstrate how to set up a virtual environment with `venv`:

1. Open a terminal and navigate to where you want to create your virtual environment
2. Once you have use the following piece of code:
```
python3 -m venv Bayesian_foundations
```

3. Using the same terminal or open a terminal activate your virtual envrionment using:
Mac command:
```
source Bayesian_foundations/bin/activate
```

Windows command:
```
Bayesian_foundations\Scripts\activate.bat
```
4. Once you've set up your virtual environment there are 4 main libraries we will work with throughout this code. They are `pandas, numpy, scipy` and `plotnine`. Run the following set of code to install those libraries:
```
pip3 install numpy scipy pandas plotnine
```

5. Next you will install several `jupyter` notebooks (or labs if you're feeling fancy) so that you can run through the code. To do that run the following set of commands:
`
pip3 install jupyterlab
`

6. Now you are ready to run the code! Navigate to the same directory you downloaded the repo's notebooks and run the following 
```
jupyter-lab
`
or for the classic notebook:
` 
jupyter-notebook
`
When running the code, make sure to select the python3 kernel which is associated virtual environment in which you installed the required libraries. This can be done by clicking on the kernel tab in the navigation menu (once you've started a notebook) or selecting the python environment located at the top right of the running notebook.