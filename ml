{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "97d6e1dd",
   "metadata": {
    "papermill": {
     "duration": 0.005754,
     "end_time": "2023-09-01T09:36:34.078620",
     "exception": false,
     "start_time": "2023-09-01T09:36:34.072866",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**This notebook is an exercise in the [Introduction to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) course.  You can reference the tutorial at [this link](https://www.kaggle.com/dansbecker/random-forests).**\n",
    "\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f0a6f45a",
   "metadata": {
    "papermill": {
     "duration": 0.004829,
     "end_time": "2023-09-01T09:36:34.088880",
     "exception": false,
     "start_time": "2023-09-01T09:36:34.084051",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Recap\n",
    "Here's the code you've written so far."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4b30c090",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:36:34.101254Z",
     "iopub.status.busy": "2023-09-01T09:36:34.100806Z",
     "iopub.status.idle": "2023-09-01T09:36:36.364203Z",
     "shell.execute_reply": "2023-09-01T09:36:36.363009Z"
    },
    "papermill": {
     "duration": 2.273129,
     "end_time": "2023-09-01T09:36:36.367147",
     "exception": false,
     "start_time": "2023-09-01T09:36:34.094018",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation MAE when not specifying max_leaf_nodes: 29,653\n",
      "Validation MAE for best value of max_leaf_nodes: 27,283\n",
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
    "print(\"Validation MAE when not specifying max_leaf_nodes: {:,.0f}\".format(val_mae))\n",
    "\n",
    "# Using best value for max_leaf_nodes\n",
    "iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)\n",
    "iowa_model.fit(train_X, train_y)\n",
    "val_predictions = iowa_model.predict(val_X)\n",
    "val_mae = mean_absolute_error(val_predictions, val_y)\n",
    "print(\"Validation MAE for best value of max_leaf_nodes: {:,.0f}\".format(val_mae))\n",
    "\n",
    "\n",
    "# Set up code checking\n",
    "from learntools.core import binder\n",
    "binder.bind(globals())\n",
    "from learntools.machine_learning.ex6 import *\n",
    "print(\"\\nSetup complete\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f12ddd28",
   "metadata": {
    "papermill": {
     "duration": 0.005012,
     "end_time": "2023-09-01T09:36:36.377891",
     "exception": false,
     "start_time": "2023-09-01T09:36:36.372879",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Exercises\n",
    "Data science isn't always this easy. But replacing the decision tree with a Random Forest is going to be an easy win."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cfc16ced",
   "metadata": {
    "papermill": {
     "duration": 0.005014,
     "end_time": "2023-09-01T09:36:36.388209",
     "exception": false,
     "start_time": "2023-09-01T09:36:36.383195",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "## Step 1: Use a Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "3badbdb2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:36:36.401677Z",
     "iopub.status.busy": "2023-09-01T09:36:36.400969Z",
     "iopub.status.idle": "2023-09-01T09:36:37.097226Z",
     "shell.execute_reply": "2023-09-01T09:36:37.096088Z"
    },
    "papermill": {
     "duration": 0.7065,
     "end_time": "2023-09-01T09:36:37.099991",
     "exception": false,
     "start_time": "2023-09-01T09:36:36.393491",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation MAE for Random Forest Model: 22227.48086627528\n"
     ]
    },
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"outcomeType\": 1, \"valueTowardsCompletion\": 1.0, \"interactionType\": 1, \"questionType\": 2, \"questionId\": \"1_CheckRfScore\", \"learnToolsVersion\": \"0.3.4\", \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\"}}, \"*\")"
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
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "# Define the model. Set random_state to 1\n",
    "\n",
    "____\n",
    "rf_model = RandomForestRegressor()\n",
    "\n",
    "rf_model.fit(train_X, train_y)\n",
    "\n",
    "# Calculate the mean absolute error of your Random Forest model on the validation data\n",
    "rf_val_predictions = rf_model.predict(val_X)\n",
    "rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)\n",
    "\n",
    "\n",
    "\n",
    "print(\"Validation MAE for Random Forest Model: {}\".format(rf_val_mae))\n",
    "\n",
    "# Check your answer\n",
    "step_1.check()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "91101285",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:36:37.119003Z",
     "iopub.status.busy": "2023-09-01T09:36:37.118196Z",
     "iopub.status.idle": "2023-09-01T09:36:37.129406Z",
     "shell.execute_reply": "2023-09-01T09:36:37.128615Z"
    },
    "papermill": {
     "duration": 0.023782,
     "end_time": "2023-09-01T09:36:37.132189",
     "exception": false,
     "start_time": "2023-09-01T09:36:37.108407",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 2, \"questionType\": 2, \"questionId\": \"1_CheckRfScore\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
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
       "<span style=\"color:#3366cc\">Hint:</span> Review the code above with a DecisionTreeRegressor. Use the RandomForestRegressor instead"
      ],
      "text/plain": [
       "Hint: Review the code above with a DecisionTreeRegressor. Use the RandomForestRegressor instead"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 3, \"questionType\": 2, \"questionId\": \"1_CheckRfScore\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
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
       "rf_model = RandomForestRegressor()\n",
       "\n",
       "# fit your model\n",
       "rf_model.fit(train_X, train_y)\n",
       "\n",
       "# Calculate the mean absolute error of your Random Forest model on the validation data\n",
       "rf_val_predictions = rf_model.predict(val_X)\n",
       "rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)\n",
       "\n",
       "```"
      ],
      "text/plain": [
       "Solution: \n",
       "```python\n",
       "rf_model = RandomForestRegressor()\n",
       "\n",
       "# fit your model\n",
       "rf_model.fit(train_X, train_y)\n",
       "\n",
       "# Calculate the mean absolute error of your Random Forest model on the validation data\n",
       "rf_val_predictions = rf_model.predict(val_X)\n",
       "rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)\n",
       "\n",
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
    "step_1.solution()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f8ee1dd7",
   "metadata": {
    "papermill": {
     "duration": 0.007382,
     "end_time": "2023-09-01T09:36:37.148636",
     "exception": false,
     "start_time": "2023-09-01T09:36:37.141254",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "So far, you have followed specific instructions at each step of your project. This helped learn key ideas and build your first model, but now you know enough to try things on your own. \n",
    "\n",
    "Machine Learning competitions are a great way to try your own ideas and learn more as you independently navigate a machine learning project. \n",
    "\n",
    "# Keep Going\n",
    "\n",
    "You are ready for **[Machine Learning Competitions](https://www.kaggle.com/alexisbcook/machine-learning-competitions).**\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "72d6d050",
   "metadata": {
    "papermill": {
     "duration": 0.006146,
     "end_time": "2023-09-01T09:36:37.161188",
     "exception": false,
     "start_time": "2023-09-01T09:36:37.155042",
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
   "duration": 7.408896,
   "end_time": "2023-09-01T09:36:37.789258",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-09-01T09:36:30.380362",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
