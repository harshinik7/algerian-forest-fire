{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "0ba8a28a",
   "metadata": {
    "papermill": {
     "duration": 0.007657,
     "end_time": "2023-09-01T09:27:40.170259",
     "exception": false,
     "start_time": "2023-09-01T09:27:40.162602",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**This notebook is an exercise in the [Introduction to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) course.  You can reference the tutorial at [this link](https://www.kaggle.com/dansbecker/underfitting-and-overfitting).**\n",
    "\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "70169bfa",
   "metadata": {
    "papermill": {
     "duration": 0.006054,
     "end_time": "2023-09-01T09:27:40.182993",
     "exception": false,
     "start_time": "2023-09-01T09:27:40.176939",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Recap\n",
    "You've built your first model, and now it's time to optimize the size of the tree to make better predictions. Run this cell to set up your coding environment where the previous step left off."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ad0f3c0b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:27:40.198778Z",
     "iopub.status.busy": "2023-09-01T09:27:40.197853Z",
     "iopub.status.idle": "2023-09-01T09:27:42.982578Z",
     "shell.execute_reply": "2023-09-01T09:27:42.981657Z"
    },
    "papermill": {
     "duration": 2.796477,
     "end_time": "2023-09-01T09:27:42.985891",
     "exception": false,
     "start_time": "2023-09-01T09:27:40.189414",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation MAE: 29,653\n",
      "\n",
      "Setup complete\n"
     ]
    }
   ],
   "source": [
    "# Code you have previously used to load data\n",
    "import pandas as pd\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "\n",
    "\n",
    "# Path of the file to read\n",
    "iowa_file_path = '../input/home-data-for-ml-course/train.csv'\n",
    "\n",
    "home_data = pd.read_csv(iowa_file_path)\n",
    "# Create target object and call it y\n",
    "y = home_data.SalePrice\n",
    "# Create X\n",
    "features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']\n",
    "X = home_data[features]\n",
    "\n",
    "# Split into validation and training data\n",
    "train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)\n",
    "\n",
    "# Specify Model\n",
    "iowa_model = DecisionTreeRegressor(random_state=1)\n",
    "# Fit Model\n",
    "iowa_model.fit(train_X, train_y)\n",
    "\n",
    "# Make validation predictions and calculate mean absolute error\n",
    "val_predictions = iowa_model.predict(val_X)\n",
    "val_mae = mean_absolute_error(val_predictions, val_y)\n",
    "print(\"Validation MAE: {:,.0f}\".format(val_mae))\n",
    "\n",
    "# Set up code checking\n",
    "from learntools.core import binder\n",
    "binder.bind(globals())\n",
    "from learntools.machine_learning.ex5 import *\n",
    "print(\"\\nSetup complete\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cfb043ff",
   "metadata": {
    "papermill": {
     "duration": 0.006934,
     "end_time": "2023-09-01T09:27:43.002401",
     "exception": false,
     "start_time": "2023-09-01T09:27:42.995467",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Exercises\n",
    "You could write the function `get_mae` yourself. For now, we'll supply it. This is the same function you read about in the previous lesson. Just run the cell below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "67e051a8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:27:43.017100Z",
     "iopub.status.busy": "2023-09-01T09:27:43.016475Z",
     "iopub.status.idle": "2023-09-01T09:27:43.022892Z",
     "shell.execute_reply": "2023-09-01T09:27:43.021914Z"
    },
    "papermill": {
     "duration": 0.018086,
     "end_time": "2023-09-01T09:27:43.026892",
     "exception": false,
     "start_time": "2023-09-01T09:27:43.008806",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):\n",
    "    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)\n",
    "    model.fit(train_X, train_y)\n",
    "    preds_val = model.predict(val_X)\n",
    "    mae = mean_absolute_error(val_y, preds_val)\n",
    "    return(mae)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "57a20ad6",
   "metadata": {
    "papermill": {
     "duration": 0.006818,
     "end_time": "2023-09-01T09:27:43.042809",
     "exception": false,
     "start_time": "2023-09-01T09:27:43.035991",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Step 1: Compare Different Tree Sizes\n",
    "Write a loop that tries the following values for *max_leaf_nodes* from a set of possible values.\n",
    "\n",
    "Call the *get_mae* function on each value of max_leaf_nodes. Store the output in some way that allows you to select the value of `max_leaf_nodes` that gives the most accurate model on your data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3ed457b6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:27:43.060567Z",
     "iopub.status.busy": "2023-09-01T09:27:43.060012Z",
     "iopub.status.idle": "2023-09-01T09:27:43.074960Z",
     "shell.execute_reply": "2023-09-01T09:27:43.073591Z"
    },
    "papermill": {
     "duration": 0.027201,
     "end_time": "2023-09-01T09:27:43.078941",
     "exception": false,
     "start_time": "2023-09-01T09:27:43.051740",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"outcomeType\": 4, \"interactionType\": 1, \"questionType\": 1, \"questionId\": \"1_BestTreeSize\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\"}}, \"*\")"
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
       "<span style=\"color:#ccaa33\">Check:</span> When you've updated the starter code, `check()` will tell you whether your code is correct. You need to update the code that creates variable `best_tree_size`"
      ],
      "text/plain": [
       "Check: When you've updated the starter code, `check()` will tell you whether your code is correct. You need to update the code that creates variable `best_tree_size`"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]\n",
    "# Write loop to find the ideal tree size from candidate_max_leaf_nodes\n",
    "_\n",
    "\n",
    "# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)\n",
    "best_tree_size = ____\n",
    "\n",
    "# Check your answer\n",
    "step_1.check()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "764e4ba2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:27:43.097416Z",
     "iopub.status.busy": "2023-09-01T09:27:43.096382Z",
     "iopub.status.idle": "2023-09-01T09:27:43.102467Z",
     "shell.execute_reply": "2023-09-01T09:27:43.100860Z"
    },
    "papermill": {
     "duration": 0.01985,
     "end_time": "2023-09-01T09:27:43.106916",
     "exception": false,
     "start_time": "2023-09-01T09:27:43.087066",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# The lines below will show you a hint or the solution.\n",
    "# step_1.hint() \n",
    "# step_1.solution()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "56e5684d",
   "metadata": {
    "papermill": {
     "duration": 0.006875,
     "end_time": "2023-09-01T09:27:43.120801",
     "exception": false,
     "start_time": "2023-09-01T09:27:43.113926",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Step 2: Fit Model Using All Data\n",
    "You know the best tree size. If you were going to deploy this model in practice, you would make it even more accurate by using all of the data and keeping that tree size.  That is, you don't need to hold out the validation data now that you've made all your modeling decisions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "970c288b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:27:43.137274Z",
     "iopub.status.busy": "2023-09-01T09:27:43.136820Z",
     "iopub.status.idle": "2023-09-01T09:27:43.147513Z",
     "shell.execute_reply": "2023-09-01T09:27:43.146125Z"
    },
    "papermill": {
     "duration": 0.022314,
     "end_time": "2023-09-01T09:27:43.150577",
     "exception": false,
     "start_time": "2023-09-01T09:27:43.128263",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"outcomeType\": 2, \"failureMessage\": \"You still need to define the following variables: `final_model`\", \"interactionType\": 1, \"questionType\": 2, \"questionId\": \"2_FitModelWithAllData\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"exceptionClass\": \"\", \"trace\": \"\"}}, \"*\")"
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
       "<span style=\"color:#cc3333\">Incorrect:</span> You still need to define the following variables: `final_model`"
      ],
      "text/plain": [
       "Incorrect: You still need to define the following variables: `final_model`"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Fill in argument to make optimal size and uncomment\n",
    "# final_model = DecisionTreeRegressor(____)\n",
    "\n",
    "# fit the final model and uncomment the next two lines\n",
    "# final_model.fit(____, ____)\n",
    "\n",
    "# Check your answer\n",
    "step_2.check()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "4bc7f31a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:27:43.169270Z",
     "iopub.status.busy": "2023-09-01T09:27:43.167856Z",
     "iopub.status.idle": "2023-09-01T09:27:43.174065Z",
     "shell.execute_reply": "2023-09-01T09:27:43.173148Z"
    },
    "papermill": {
     "duration": 0.018403,
     "end_time": "2023-09-01T09:27:43.176683",
     "exception": false,
     "start_time": "2023-09-01T09:27:43.158280",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# step_2.hint()\n",
    "# step_2.solution()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c2972d17",
   "metadata": {
    "papermill": {
     "duration": 0.007601,
     "end_time": "2023-09-01T09:27:43.191833",
     "exception": false,
     "start_time": "2023-09-01T09:27:43.184232",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "You've tuned this model and improved your results. But we are still using Decision Tree models, which are not very sophisticated by modern machine learning standards. In the next step you will learn to use Random Forests to improve your models even more.\n",
    "\n",
    "# Keep Going\n",
    "\n",
    "You are ready for **[Random Forests](https://www.kaggle.com/dansbecker/random-forests).**\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f31f05e7",
   "metadata": {
    "papermill": {
     "duration": 0.007813,
     "end_time": "2023-09-01T09:27:43.207150",
     "exception": false,
     "start_time": "2023-09-01T09:27:43.199337",
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
   "duration": 7.929542,
   "end_time": "2023-09-01T09:27:43.939520",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-09-01T09:27:36.009978",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
