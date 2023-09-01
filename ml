{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9ba51234",
   "metadata": {
    "papermill": {
     "duration": 0.009739,
     "end_time": "2023-09-01T09:06:36.631222",
     "exception": false,
     "start_time": "2023-09-01T09:06:36.621483",
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
   "id": "0fe8c336",
   "metadata": {
    "papermill": {
     "duration": 0.008325,
     "end_time": "2023-09-01T09:06:36.648392",
     "exception": false,
     "start_time": "2023-09-01T09:06:36.640067",
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
   "id": "0306e1c3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:06:36.668345Z",
     "iopub.status.busy": "2023-09-01T09:06:36.667620Z",
     "iopub.status.idle": "2023-09-01T09:06:39.095865Z",
     "shell.execute_reply": "2023-09-01T09:06:39.094587Z"
    },
    "papermill": {
     "duration": 2.441856,
     "end_time": "2023-09-01T09:06:39.098820",
     "exception": false,
     "start_time": "2023-09-01T09:06:36.656964",
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
   "id": "cb2428c3",
   "metadata": {
    "papermill": {
     "duration": 0.008804,
     "end_time": "2023-09-01T09:06:39.117070",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.108266",
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
   "id": "9f0c12d1",
   "metadata": {
    "collapsed": true,
    "execution": {
     "iopub.execute_input": "2023-09-01T09:06:39.137369Z",
     "iopub.status.busy": "2023-09-01T09:06:39.136804Z",
     "iopub.status.idle": "2023-09-01T09:06:39.141678Z",
     "shell.execute_reply": "2023-09-01T09:06:39.140502Z"
    },
    "jupyter": {
     "outputs_hidden": true
    },
    "papermill": {
     "duration": 0.018001,
     "end_time": "2023-09-01T09:06:39.144024",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.126023",
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
   "id": "92d8483c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:06:39.164536Z",
     "iopub.status.busy": "2023-09-01T09:06:39.163818Z",
     "iopub.status.idle": "2023-09-01T09:06:39.180348Z",
     "shell.execute_reply": "2023-09-01T09:06:39.179229Z"
    },
    "papermill": {
     "duration": 0.02962,
     "end_time": "2023-09-01T09:06:39.182790",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.153170",
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
   "id": "2e8c4b45",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:06:39.202910Z",
     "iopub.status.busy": "2023-09-01T09:06:39.202290Z",
     "iopub.status.idle": "2023-09-01T09:06:39.215363Z",
     "shell.execute_reply": "2023-09-01T09:06:39.214216Z"
    },
    "papermill": {
     "duration": 0.025851,
     "end_time": "2023-09-01T09:06:39.217651",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.191800",
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
   "id": "61926ad4",
   "metadata": {
    "papermill": {
     "duration": 0.009505,
     "end_time": "2023-09-01T09:06:39.236728",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.227223",
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
   "id": "175fe990",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:06:39.258256Z",
     "iopub.status.busy": "2023-09-01T09:06:39.257578Z",
     "iopub.status.idle": "2023-09-01T09:06:39.276784Z",
     "shell.execute_reply": "2023-09-01T09:06:39.275556Z"
    },
    "papermill": {
     "duration": 0.032671,
     "end_time": "2023-09-01T09:06:39.279022",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.246351",
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
   "id": "c3078276",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:06:39.304307Z",
     "iopub.status.busy": "2023-09-01T09:06:39.303629Z",
     "iopub.status.idle": "2023-09-01T09:06:39.317768Z",
     "shell.execute_reply": "2023-09-01T09:06:39.316430Z"
    },
    "papermill": {
     "duration": 0.030612,
     "end_time": "2023-09-01T09:06:39.320460",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.289848",
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
   "id": "ed563a4f",
   "metadata": {
    "papermill": {
     "duration": 0.011272,
     "end_time": "2023-09-01T09:06:39.342789",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.331517",
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
   "id": "e49cc61b",
   "metadata": {
    "collapsed": true,
    "execution": {
     "iopub.execute_input": "2023-09-01T09:06:39.367497Z",
     "iopub.status.busy": "2023-09-01T09:06:39.366823Z",
     "iopub.status.idle": "2023-09-01T09:06:39.372239Z",
     "shell.execute_reply": "2023-09-01T09:06:39.370936Z"
    },
    "jupyter": {
     "outputs_hidden": true
    },
    "papermill": {
     "duration": 0.021144,
     "end_time": "2023-09-01T09:06:39.374745",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.353601",
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
   "id": "8c145cb4",
   "metadata": {
    "papermill": {
     "duration": 0.010496,
     "end_time": "2023-09-01T09:06:39.396603",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.386107",
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
   "id": "6c6a1733",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:06:39.420672Z",
     "iopub.status.busy": "2023-09-01T09:06:39.420168Z",
     "iopub.status.idle": "2023-09-01T09:06:39.443050Z",
     "shell.execute_reply": "2023-09-01T09:06:39.442062Z"
    },
    "papermill": {
     "duration": 0.037734,
     "end_time": "2023-09-01T09:06:39.445257",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.407523",
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
   "id": "0c238fa8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:06:39.471298Z",
     "iopub.status.busy": "2023-09-01T09:06:39.470124Z",
     "iopub.status.idle": "2023-09-01T09:06:39.484557Z",
     "shell.execute_reply": "2023-09-01T09:06:39.483514Z"
    },
    "papermill": {
     "duration": 0.029826,
     "end_time": "2023-09-01T09:06:39.486888",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.457062",
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
   "id": "7c9e6d35",
   "metadata": {
    "papermill": {
     "duration": 0.01226,
     "end_time": "2023-09-01T09:06:39.511942",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.499682",
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
   "id": "d0817f14",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:06:39.539629Z",
     "iopub.status.busy": "2023-09-01T09:06:39.538887Z",
     "iopub.status.idle": "2023-09-01T09:06:39.554037Z",
     "shell.execute_reply": "2023-09-01T09:06:39.552922Z"
    },
    "papermill": {
     "duration": 0.031643,
     "end_time": "2023-09-01T09:06:39.556203",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.524560",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[208500. 181500. 223500. ... 266500. 142125. 147500.]\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"outcomeType\": 1, \"valueTowardsCompletion\": 0.25, \"interactionType\": 1, \"questionType\": 2, \"questionId\": \"4_MakePredictions\", \"learnToolsVersion\": \"0.3.4\", \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\"}}, \"*\")"
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
    "predictions = iowa_model.predict(X)\n",
    "print(predictions)\n",
    "\n",
    "# Check your answer\n",
    "step_4.check()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6718f1de",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:06:39.583277Z",
     "iopub.status.busy": "2023-09-01T09:06:39.582646Z",
     "iopub.status.idle": "2023-09-01T09:06:39.596002Z",
     "shell.execute_reply": "2023-09-01T09:06:39.595204Z"
    },
    "papermill": {
     "duration": 0.030031,
     "end_time": "2023-09-01T09:06:39.598803",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.568772",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 2, \"questionType\": 2, \"questionId\": \"4_MakePredictions\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
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
       "<span style=\"color:#3366cc\">Hint:</span> Use `iowa_model.predict` with an argument holding the data to predict with."
      ],
      "text/plain": [
       "Hint: Use `iowa_model.predict` with an argument holding the data to predict with."
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 3, \"questionType\": 2, \"questionId\": \"4_MakePredictions\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
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
       "iowa_model.predict(X)\n",
       "```"
      ],
      "text/plain": [
       "Solution: \n",
       "```python\n",
       "iowa_model.predict(X)\n",
       "```"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    " step_4.hint()\n",
    " step_4.solution()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5a102e29",
   "metadata": {
    "papermill": {
     "duration": 0.013017,
     "end_time": "2023-09-01T09:06:39.625265",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.612248",
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
   "id": "f82a2945",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:06:39.653623Z",
     "iopub.status.busy": "2023-09-01T09:06:39.653220Z",
     "iopub.status.idle": "2023-09-01T09:06:39.658089Z",
     "shell.execute_reply": "2023-09-01T09:06:39.656852Z"
    },
    "papermill": {
     "duration": 0.022229,
     "end_time": "2023-09-01T09:06:39.660722",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.638493",
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
   "id": "048eedd3",
   "metadata": {
    "papermill": {
     "duration": 0.013082,
     "end_time": "2023-09-01T09:06:39.688247",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.675165",
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
   "id": "a9a0824c",
   "metadata": {
    "papermill": {
     "duration": 0.0132,
     "end_time": "2023-09-01T09:06:39.714906",
     "exception": false,
     "start_time": "2023-09-01T09:06:39.701706",
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
   "duration": 7.559532,
   "end_time": "2023-09-01T09:06:40.351566",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-09-01T09:06:32.792034",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
