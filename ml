{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "db5e61b7",
   "metadata": {
    "papermill": {
     "duration": 0.006856,
     "end_time": "2023-09-01T09:05:16.213694",
     "exception": false,
     "start_time": "2023-09-01T09:05:16.206838",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**This notebook is an exercise in the [Introduction to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) course.  You can reference the tutorial at [this link](https://www.kaggle.com/dansbecker/your-first-machine-learning-model).**\n",
    "\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d0c2665d",
   "metadata": {
    "papermill": {
     "duration": 0.006779,
     "end_time": "2023-09-01T09:05:16.227808",
     "exception": false,
     "start_time": "2023-09-01T09:05:16.221029",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Recap\n",
    "So far, you have loaded your data and reviewed it with the following code. Run this cell to set up your coding environment where the previous step left off."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8e87b196",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:05:16.242388Z",
     "iopub.status.busy": "2023-09-01T09:05:16.241872Z",
     "iopub.status.idle": "2023-09-01T09:05:18.462306Z",
     "shell.execute_reply": "2023-09-01T09:05:18.461121Z"
    },
    "papermill": {
     "duration": 2.230604,
     "end_time": "2023-09-01T09:05:18.464708",
     "exception": false,
     "start_time": "2023-09-01T09:05:16.234104",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Setup Complete\n"
     ]
    }
   ],
   "source": [
    "# Code you have previously used to load data\n",
    "import pandas as pd\n",
    "\n",
    "# Path of the file to read\n",
    "iowa_file_path = '../input/home-data-for-ml-course/train.csv'\n",
    "\n",
    "home_data = pd.read_csv(iowa_file_path)\n",
    "\n",
    "# Set up code checking\n",
    "from learntools.core import binder\n",
    "binder.bind(globals())\n",
    "from learntools.machine_learning.ex3 import *\n",
    "\n",
    "print(\"Setup Complete\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "35c3f93c",
   "metadata": {
    "papermill": {
     "duration": 0.006156,
     "end_time": "2023-09-01T09:05:18.477176",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.471020",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Exercises\n",
    "\n",
    "## Step 1: Specify Prediction Target\n",
    "Select the target variable, which corresponds to the sales price. Save this to a new variable called `y`. You'll need to print a list of the columns to find the name of the column you need.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "ef6136ff",
   "metadata": {
    "collapsed": true,
    "execution": {
     "iopub.execute_input": "2023-09-01T09:05:18.491686Z",
     "iopub.status.busy": "2023-09-01T09:05:18.490714Z",
     "iopub.status.idle": "2023-09-01T09:05:18.495201Z",
     "shell.execute_reply": "2023-09-01T09:05:18.494289Z"
    },
    "jupyter": {
     "outputs_hidden": true
    },
    "papermill": {
     "duration": 0.013854,
     "end_time": "2023-09-01T09:05:18.497232",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.483378",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# print the list of columns in the dataset to find the name of the prediction target\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "56dfeb98",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:05:18.513143Z",
     "iopub.status.busy": "2023-09-01T09:05:18.512473Z",
     "iopub.status.idle": "2023-09-01T09:05:18.527782Z",
     "shell.execute_reply": "2023-09-01T09:05:18.526509Z"
    },
    "papermill": {
     "duration": 0.026032,
     "end_time": "2023-09-01T09:05:18.530494",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.504462",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"outcomeType\": 1, \"valueTowardsCompletion\": 0.25, \"interactionType\": 1, \"questionType\": 2, \"questionId\": \"1_SetTarget\", \"learnToolsVersion\": \"0.3.4\", \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\"}}, \"*\")"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/markdown": [
       "<span style=\"color:#33cc33\">Correct</span>"
      ],
      "text/plain": [
       "Correct"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "y = home_data.SalePrice\n",
    "\n",
    "# Check your answer\n",
    "step_1.check()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b07b48ad",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:05:18.545065Z",
     "iopub.status.busy": "2023-09-01T09:05:18.544702Z",
     "iopub.status.idle": "2023-09-01T09:05:18.556478Z",
     "shell.execute_reply": "2023-09-01T09:05:18.555479Z"
    },
    "papermill": {
     "duration": 0.021556,
     "end_time": "2023-09-01T09:05:18.558607",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.537051",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 2, \"questionType\": 2, \"questionId\": \"1_SetTarget\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/markdown": [
       "<span style=\"color:#3366cc\">Hint:</span> Use `print(home_data.columns)`. The column you want is at the end of the list. Use the dot notation to pull out this column from the DataFrame"
      ],
      "text/plain": [
       "Hint: Use `print(home_data.columns)`. The column you want is at the end of the list. Use the dot notation to pull out this column from the DataFrame"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 3, \"questionType\": 2, \"questionId\": \"1_SetTarget\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/markdown": [
       "<span style=\"color:#33cc99\">Solution:</span> \n",
       "```python\n",
       "y = home_data.SalePrice\n",
       "```"
      ],
      "text/plain": [
       "Solution: \n",
       "```python\n",
       "y = home_data.SalePrice\n",
       "```"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# The lines below will show you a hint or the solution.\n",
    "step_1.hint() \n",
    "step_1.solution()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f1774067",
   "metadata": {
    "papermill": {
     "duration": 0.007197,
     "end_time": "2023-09-01T09:05:18.572757",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.565560",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Step 2: Create X\n",
    "Now you will create a DataFrame called `X` holding the predictive features.\n",
    "\n",
    "Since you want only some columns from the original data, you'll first create a list with the names of the columns you want in `X`.\n",
    "\n",
    "You'll use just the following columns in the list (you can copy and paste the whole list to save some typing, though you'll still need to add quotes):\n",
    "  * LotArea\n",
    "  * YearBuilt\n",
    "  * 1stFlrSF\n",
    "  * 2ndFlrSF\n",
    "  * FullBath\n",
    "  * BedroomAbvGr\n",
    "  * TotRmsAbvGrd\n",
    "\n",
    "After you've created that list of features, use it to create the DataFrame that you'll use to fit the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "127e8b76",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:05:18.588813Z",
     "iopub.status.busy": "2023-09-01T09:05:18.588132Z",
     "iopub.status.idle": "2023-09-01T09:05:18.606345Z",
     "shell.execute_reply": "2023-09-01T09:05:18.605297Z"
    },
    "papermill": {
     "duration": 0.029011,
     "end_time": "2023-09-01T09:05:18.608568",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.579557",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"outcomeType\": 1, \"valueTowardsCompletion\": 0.25, \"interactionType\": 1, \"questionType\": 2, \"questionId\": \"2_SelectPredictionData\", \"learnToolsVersion\": \"0.3.4\", \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\"}}, \"*\")"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/markdown": [
       "<span style=\"color:#33cc33\">Correct</span>"
      ],
      "text/plain": [
       "Correct"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Create the list of features below\n",
    "\n",
    "\n",
    "feature_names = [\"LotArea\", \"YearBuilt\", \"1stFlrSF\", \"2ndFlrSF\",\n",
    "                      \"FullBath\", \"BedroomAbvGr\", \"TotRmsAbvGrd\"]\n",
    "\n",
    "X=home_data[feature_names]\n",
    "\n",
    "# Check your answer\n",
    "step_2.check()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "62cdd9f5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:05:18.624693Z",
     "iopub.status.busy": "2023-09-01T09:05:18.624300Z",
     "iopub.status.idle": "2023-09-01T09:05:18.636414Z",
     "shell.execute_reply": "2023-09-01T09:05:18.635273Z"
    },
    "papermill": {
     "duration": 0.022721,
     "end_time": "2023-09-01T09:05:18.638640",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.615919",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 2, \"questionType\": 2, \"questionId\": \"2_SelectPredictionData\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/markdown": [
       "<span style=\"color:#3366cc\">Hint:</span> Capitalization and spelling are important when specifying variable names. Use the brackets notation when specifying data for X."
      ],
      "text/plain": [
       "Hint: Capitalization and spelling are important when specifying variable names. Use the brackets notation when specifying data for X."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 3, \"questionType\": 2, \"questionId\": \"2_SelectPredictionData\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/markdown": [
       "<span style=\"color:#33cc99\">Solution:</span> \n",
       "```python\n",
       "feature_names = [\"LotArea\", \"YearBuilt\", \"1stFlrSF\", \"2ndFlrSF\",\n",
       "                      \"FullBath\", \"BedroomAbvGr\", \"TotRmsAbvGrd\"]\n",
       "\n",
       "X=home_data[feature_names]\n",
       "```"
      ],
      "text/plain": [
       "Solution: \n",
       "```python\n",
       "feature_names = [\"LotArea\", \"YearBuilt\", \"1stFlrSF\", \"2ndFlrSF\",\n",
       "                      \"FullBath\", \"BedroomAbvGr\", \"TotRmsAbvGrd\"]\n",
       "\n",
       "X=home_data[feature_names]\n",
       "```"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    " step_2.hint()\n",
    " step_2.solution()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "29de66a1",
   "metadata": {
    "papermill": {
     "duration": 0.00739,
     "end_time": "2023-09-01T09:05:18.653784",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.646394",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Review Data\n",
    "Before building a model, take a quick look at **X** to verify it looks sensible"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8bb21076",
   "metadata": {
    "collapsed": true,
    "execution": {
     "iopub.execute_input": "2023-09-01T09:05:18.671465Z",
     "iopub.status.busy": "2023-09-01T09:05:18.670699Z",
     "iopub.status.idle": "2023-09-01T09:05:18.675195Z",
     "shell.execute_reply": "2023-09-01T09:05:18.674398Z"
    },
    "jupyter": {
     "outputs_hidden": true
    },
    "papermill": {
     "duration": 0.016082,
     "end_time": "2023-09-01T09:05:18.677640",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.661558",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Review data\n",
    "# print description or statistics from X\n",
    "#print(_)\n",
    "\n",
    "# print the top few lines\n",
    "#print(_)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "72e28c77",
   "metadata": {
    "papermill": {
     "duration": 0.007477,
     "end_time": "2023-09-01T09:05:18.693428",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.685951",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Step 3: Specify and Fit Model\n",
    "Create a `DecisionTreeRegressor` and save it iowa_model. Ensure you've done the relevant import from sklearn to run this command.\n",
    "\n",
    "Then fit the model you just created using the data in `X` and `y` that you saved above."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c931e312",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:05:18.711140Z",
     "iopub.status.busy": "2023-09-01T09:05:18.710314Z",
     "iopub.status.idle": "2023-09-01T09:05:18.730685Z",
     "shell.execute_reply": "2023-09-01T09:05:18.729533Z"
    },
    "papermill": {
     "duration": 0.032108,
     "end_time": "2023-09-01T09:05:18.733145",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.701037",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"outcomeType\": 1, \"valueTowardsCompletion\": 0.25, \"interactionType\": 1, \"questionType\": 2, \"questionId\": \"3_CreateModel\", \"learnToolsVersion\": \"0.3.4\", \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\"}}, \"*\")"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/markdown": [
       "<span style=\"color:#33cc33\">Correct</span>"
      ],
      "text/plain": [
       "Correct"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# from _ import _\n",
    "#specify the model. \n",
    "#For model reproducibility, set a numeric value for random_state when specifying the model\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "iowa_model = DecisionTreeRegressor(random_state=1)\n",
    "iowa_model.fit(X, y)\n",
    "\n",
    "# Check your answer\n",
    "step_3.check()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "2281c2e3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:05:18.752513Z",
     "iopub.status.busy": "2023-09-01T09:05:18.752104Z",
     "iopub.status.idle": "2023-09-01T09:05:18.764589Z",
     "shell.execute_reply": "2023-09-01T09:05:18.763516Z"
    },
    "papermill": {
     "duration": 0.025426,
     "end_time": "2023-09-01T09:05:18.767041",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.741615",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 2, \"questionType\": 2, \"questionId\": \"3_CreateModel\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/markdown": [
       "<span style=\"color:#3366cc\">Hint:</span> Include `random_state` when specifying model. Data is specified when fitting it."
      ],
      "text/plain": [
       "Hint: Include `random_state` when specifying model. Data is specified when fitting it."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 3, \"questionType\": 2, \"questionId\": \"3_CreateModel\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/markdown": [
       "<span style=\"color:#33cc99\">Solution:</span> \n",
       "```python\n",
       "from sklearn.tree import DecisionTreeRegressor\n",
       "iowa_model = DecisionTreeRegressor(random_state=1)\n",
       "iowa_model.fit(X, y)\n",
       "```"
      ],
      "text/plain": [
       "Solution: \n",
       "```python\n",
       "from sklearn.tree import DecisionTreeRegressor\n",
       "iowa_model = DecisionTreeRegressor(random_state=1)\n",
       "iowa_model.fit(X, y)\n",
       "```"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    " step_3.hint()\n",
    " step_3.solution()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "64458c08",
   "metadata": {
    "papermill": {
     "duration": 0.009019,
     "end_time": "2023-09-01T09:05:18.784682",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.775663",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Step 4: Make Predictions\n",
    "Make predictions with the model's `predict` command using `X` as the data. Save the results to a variable called `predictions`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "1915713e",
   "metadata": {
    "collapsed": true,
    "execution": {
     "iopub.execute_input": "2023-09-01T09:05:18.803538Z",
     "iopub.status.busy": "2023-09-01T09:05:18.803107Z",
     "iopub.status.idle": "2023-09-01T09:05:18.813192Z",
     "shell.execute_reply": "2023-09-01T09:05:18.812284Z"
    },
    "jupyter": {
     "outputs_hidden": true
    },
    "papermill": {
     "duration": 0.02212,
     "end_time": "2023-09-01T09:05:18.815399",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.793279",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<learntools.core.constants.PlaceholderValue object at 0x791040dec9d0>\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"outcomeType\": 4, \"interactionType\": 1, \"questionType\": 2, \"questionId\": \"4_MakePredictions\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\"}}, \"*\")"
      ],
      "text/plain": [
       "<IPython.core.display.Javascript object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/markdown": [
       "<span style=\"color:#ccaa33\">Check:</span> When you've updated the starter code, `check()` will tell you whether your code is correct. You need to update the code that creates variables `predictions`, `iowa_model`, `X`"
      ],
      "text/plain": [
       "Check: When you've updated the starter code, `check()` will tell you whether your code is correct. You need to update the code that creates variables `predictions`, `iowa_model`, `X`"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "predictions = ____\n",
    "print(predictions)\n",
    "\n",
    "# Check your answer\n",
    "step_4.check()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cd1e7ba7",
   "metadata": {
    "collapsed": true,
    "execution": {
     "iopub.execute_input": "2023-09-01T09:05:18.837825Z",
     "iopub.status.busy": "2023-09-01T09:05:18.836964Z",
     "iopub.status.idle": "2023-09-01T09:05:18.842686Z",
     "shell.execute_reply": "2023-09-01T09:05:18.841773Z"
    },
    "jupyter": {
     "outputs_hidden": true
    },
    "papermill": {
     "duration": 0.019943,
     "end_time": "2023-09-01T09:05:18.844891",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.824948",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# step_4.hint()\n",
    "# step_4.solution()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b078ccd1",
   "metadata": {
    "papermill": {
     "duration": 0.008911,
     "end_time": "2023-09-01T09:05:18.863153",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.854242",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Think About Your Results\n",
    "\n",
    "Use the `head` method to compare the top few predictions to the actual home values (in `y`) for those same homes. Anything surprising?\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "0e19a62d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:05:18.882780Z",
     "iopub.status.busy": "2023-09-01T09:05:18.882364Z",
     "iopub.status.idle": "2023-09-01T09:05:18.887177Z",
     "shell.execute_reply": "2023-09-01T09:05:18.885961Z"
    },
    "papermill": {
     "duration": 0.017691,
     "end_time": "2023-09-01T09:05:18.889850",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.872159",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# You can write code in this cell\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c7bd2ea7",
   "metadata": {
    "papermill": {
     "duration": 0.008779,
     "end_time": "2023-09-01T09:05:18.907944",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.899165",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "It's natural to ask how accurate the model's predictions will be and how you can improve that. That will be you're next step.\n",
    "\n",
    "# Keep Going\n",
    "\n",
    "You are ready for **[Model Validation](https://www.kaggle.com/dansbecker/model-validation).**\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1467d138",
   "metadata": {
    "papermill": {
     "duration": 0.008939,
     "end_time": "2023-09-01T09:05:18.926798",
     "exception": false,
     "start_time": "2023-09-01T09:05:18.917859",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "---\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "*Have questions or comments? Visit the [course discussion forum](https://www.kaggle.com/learn/intro-to-machine-learning/discussion) to chat with other learners.*"
   ]
  }
 ],
 "kernelspec": {
  "display_name": "Python 3",
  "language": "python",
  "name": "python3"
 },
 "language_info": {
  "codemirror_mode": {
   "name": "ipython",
   "version": 3
  },
  "file_extension": ".py",
  "mimetype": "text/x-python",
  "name": "python",
  "nbconvert_exporter": "python",
  "pygments_lexer": "ipython3",
  "version": "3.6.4"
 },
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 6.685447,
   "end_time": "2023-09-01T09:05:19.558013",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-09-01T09:05:12.872566",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
