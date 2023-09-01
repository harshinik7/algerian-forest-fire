{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9071f33a",
   "metadata": {
    "papermill": {
     "duration": 0.007178,
     "end_time": "2023-09-01T09:29:21.673650",
     "exception": false,
     "start_time": "2023-09-01T09:29:21.666472",
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
   "id": "a760428e",
   "metadata": {
    "papermill": {
     "duration": 0.005993,
     "end_time": "2023-09-01T09:29:21.686253",
     "exception": false,
     "start_time": "2023-09-01T09:29:21.680260",
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
   "id": "01046e8b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:29:21.701853Z",
     "iopub.status.busy": "2023-09-01T09:29:21.701378Z",
     "iopub.status.idle": "2023-09-01T09:29:24.180224Z",
     "shell.execute_reply": "2023-09-01T09:29:24.178966Z"
    },
    "papermill": {
     "duration": 2.490348,
     "end_time": "2023-09-01T09:29:24.183482",
     "exception": false,
     "start_time": "2023-09-01T09:29:21.693134",
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
   "id": "055a3aae",
   "metadata": {
    "papermill": {
     "duration": 0.0061,
     "end_time": "2023-09-01T09:29:24.196121",
     "exception": false,
     "start_time": "2023-09-01T09:29:24.190021",
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
   "id": "a89f5f4f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:29:24.210825Z",
     "iopub.status.busy": "2023-09-01T09:29:24.210373Z",
     "iopub.status.idle": "2023-09-01T09:29:24.216324Z",
     "shell.execute_reply": "2023-09-01T09:29:24.215527Z"
    },
    "papermill": {
     "duration": 0.016087,
     "end_time": "2023-09-01T09:29:24.218522",
     "exception": false,
     "start_time": "2023-09-01T09:29:24.202435",
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
   "id": "662bbffb",
   "metadata": {
    "papermill": {
     "duration": 0.006259,
     "end_time": "2023-09-01T09:29:24.231406",
     "exception": false,
     "start_time": "2023-09-01T09:29:24.225147",
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
   "id": "b5a2e692",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:29:24.246656Z",
     "iopub.status.busy": "2023-09-01T09:29:24.245877Z",
     "iopub.status.idle": "2023-09-01T09:29:24.300084Z",
     "shell.execute_reply": "2023-09-01T09:29:24.298923Z"
    },
    "papermill": {
     "duration": 0.065138,
     "end_time": "2023-09-01T09:29:24.302991",
     "exception": false,
     "start_time": "2023-09-01T09:29:24.237853",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"outcomeType\": 1, \"valueTowardsCompletion\": 0.5, \"interactionType\": 1, \"questionType\": 1, \"questionId\": \"1_BestTreeSize\", \"learnToolsVersion\": \"0.3.4\", \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\"}}, \"*\")"
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
    "candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]\n",
    "# Write loop to find the ideal tree size from candidate_max_leaf_nodes\n",
    "_\n",
    "\n",
    "# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)\n",
    "best_tree_size = ____\n",
    "scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}\n",
    "best_tree_size = min(scores, key=scores.get)\n",
    "# Check your answer\n",
    "step_1.check()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0ab48aa0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:29:24.319387Z",
     "iopub.status.busy": "2023-09-01T09:29:24.318480Z",
     "iopub.status.idle": "2023-09-01T09:29:24.332497Z",
     "shell.execute_reply": "2023-09-01T09:29:24.331389Z"
    },
    "papermill": {
     "duration": 0.024875,
     "end_time": "2023-09-01T09:29:24.334955",
     "exception": false,
     "start_time": "2023-09-01T09:29:24.310080",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 2, \"questionType\": 1, \"questionId\": \"1_BestTreeSize\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
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
       "<span style=\"color:#3366cc\">Hint:</span> You will call get_mae in the loop. You'll need to map the names of your data structure to the names in get_mae"
      ],
      "text/plain": [
       "Hint: You will call get_mae in the loop. You'll need to map the names of your data structure to the names in get_mae"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 3, \"questionType\": 1, \"questionId\": \"1_BestTreeSize\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
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
       "# Here is a short solution with a dict comprehension.\n",
       "# The lesson gives an example of how to do this with an explicit loop.\n",
       "scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}\n",
       "best_tree_size = min(scores, key=scores.get)\n",
       "\n",
       "```"
      ],
      "text/plain": [
       "Solution: \n",
       "```python\n",
       "# Here is a short solution with a dict comprehension.\n",
       "# The lesson gives an example of how to do this with an explicit loop.\n",
       "scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}\n",
       "best_tree_size = min(scores, key=scores.get)\n",
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
    "step_1.solution()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c2017619",
   "metadata": {
    "papermill": {
     "duration": 0.007546,
     "end_time": "2023-09-01T09:29:24.350045",
     "exception": false,
     "start_time": "2023-09-01T09:29:24.342499",
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
   "id": "6e61778b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:29:24.369418Z",
     "iopub.status.busy": "2023-09-01T09:29:24.368900Z",
     "iopub.status.idle": "2023-09-01T09:29:24.389406Z",
     "shell.execute_reply": "2023-09-01T09:29:24.388201Z"
    },
    "papermill": {
     "duration": 0.032059,
     "end_time": "2023-09-01T09:29:24.391809",
     "exception": false,
     "start_time": "2023-09-01T09:29:24.359750",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"outcomeType\": 1, \"valueTowardsCompletion\": 0.5, \"interactionType\": 1, \"questionType\": 2, \"questionId\": \"2_FitModelWithAllData\", \"learnToolsVersion\": \"0.3.4\", \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\"}}, \"*\")"
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
    "# Fill in argument to make optimal size and uncomment\n",
    "# final_model = DecisionTreeRegressor(____)\n",
    "\n",
    "# fit the final model and uncomment the next two lines\n",
    "# final_model.fit(____, ____)\n",
    "final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)\n",
    "\n",
    "# fit the final model\n",
    "final_model.fit(X, y)\n",
    "# Check your answer\n",
    "step_2.check()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "9f4cf945",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:29:24.410128Z",
     "iopub.status.busy": "2023-09-01T09:29:24.409635Z",
     "iopub.status.idle": "2023-09-01T09:29:24.425048Z",
     "shell.execute_reply": "2023-09-01T09:29:24.423806Z"
    },
    "papermill": {
     "duration": 0.027869,
     "end_time": "2023-09-01T09:29:24.427659",
     "exception": false,
     "start_time": "2023-09-01T09:29:24.399790",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 2, \"questionType\": 2, \"questionId\": \"2_FitModelWithAllData\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
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
       "<span style=\"color:#3366cc\">Hint:</span> Fit with the ideal value of max_leaf_nodes. In the fit step, use all of the data in the dataset"
      ],
      "text/plain": [
       "Hint: Fit with the ideal value of max_leaf_nodes. In the fit step, use all of the data in the dataset"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 3, \"questionType\": 2, \"questionId\": \"2_FitModelWithAllData\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
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
       "# Fit the model with best_tree_size. Fill in argument to make optimal size\n",
       "final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)\n",
       "\n",
       "# fit the final model\n",
       "final_model.fit(X, y)\n",
       "```"
      ],
      "text/plain": [
       "Solution: \n",
       "```python\n",
       "# Fit the model with best_tree_size. Fill in argument to make optimal size\n",
       "final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)\n",
       "\n",
       "# fit the final model\n",
       "final_model.fit(X, y)\n",
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
   "id": "cc23ba73",
   "metadata": {
    "papermill": {
     "duration": 0.008452,
     "end_time": "2023-09-01T09:29:24.445173",
     "exception": false,
     "start_time": "2023-09-01T09:29:24.436721",
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
   "id": "ec3ac60c",
   "metadata": {
    "papermill": {
     "duration": 0.008307,
     "end_time": "2023-09-01T09:29:24.462229",
     "exception": false,
     "start_time": "2023-09-01T09:29:24.453922",
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
   "duration": 7.313267,
   "end_time": "2023-09-01T09:29:25.092603",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-09-01T09:29:17.779336",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
