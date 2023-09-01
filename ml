{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "05107292",
   "metadata": {
    "papermill": {
     "duration": 0.007363,
     "end_time": "2023-09-01T09:39:52.877521",
     "exception": false,
     "start_time": "2023-09-01T09:39:52.870158",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**This notebook is an exercise in the [Introduction to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) course.  You can reference the tutorial at [this link](https://www.kaggle.com/alexisbcook/machine-learning-competitions).**\n",
    "\n",
    "---\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7e2ae6b6",
   "metadata": {
    "papermill": {
     "duration": 0.006317,
     "end_time": "2023-09-01T09:39:52.890797",
     "exception": false,
     "start_time": "2023-09-01T09:39:52.884480",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Introduction\n",
    "\n",
    "In this exercise, you will create and submit predictions for a Kaggle competition. You can then improve your model (e.g. by adding features) to apply what you've learned and move up the leaderboard.\n",
    "\n",
    "Begin by running the code cell below to set up code checking and the filepaths for the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "35c0ba7d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:39:52.906243Z",
     "iopub.status.busy": "2023-09-01T09:39:52.905518Z",
     "iopub.status.idle": "2023-09-01T09:39:53.373671Z",
     "shell.execute_reply": "2023-09-01T09:39:53.372605Z"
    },
    "papermill": {
     "duration": 0.479116,
     "end_time": "2023-09-01T09:39:53.376510",
     "exception": false,
     "start_time": "2023-09-01T09:39:52.897394",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Set up code checking\n",
    "from learntools.core import binder\n",
    "binder.bind(globals())\n",
    "from learntools.machine_learning.ex7 import *\n",
    "\n",
    "# Set up filepaths\n",
    "import os\n",
    "if not os.path.exists(\"../input/train.csv\"):\n",
    "    os.symlink(\"../input/home-data-for-ml-course/train.csv\", \"../input/train.csv\")  \n",
    "    os.symlink(\"../input/home-data-for-ml-course/test.csv\", \"../input/test.csv\") "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1f1520da",
   "metadata": {
    "papermill": {
     "duration": 0.006332,
     "end_time": "2023-09-01T09:39:53.389818",
     "exception": false,
     "start_time": "2023-09-01T09:39:53.383486",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Here's some of the code you've written so far. Start by running it again."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1df1ba3b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:39:53.405373Z",
     "iopub.status.busy": "2023-09-01T09:39:53.404774Z",
     "iopub.status.idle": "2023-09-01T09:39:55.926963Z",
     "shell.execute_reply": "2023-09-01T09:39:55.925705Z"
    },
    "papermill": {
     "duration": 2.533463,
     "end_time": "2023-09-01T09:39:55.929935",
     "exception": false,
     "start_time": "2023-09-01T09:39:53.396472",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Validation MAE for Random Forest Model: 21,857\n"
     ]
    }
   ],
   "source": [
    "# Import helpful libraries\n",
    "import pandas as pd\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "# Load the data, and separate the target\n",
    "iowa_file_path = '../input/train.csv'\n",
    "home_data = pd.read_csv(iowa_file_path)\n",
    "y = home_data.SalePrice\n",
    "\n",
    "# Create X (After completing the exercise, you can return to modify this line!)\n",
    "features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']\n",
    "\n",
    "# Select columns corresponding to features, and preview the data\n",
    "X = home_data[features]\n",
    "X.head()\n",
    "\n",
    "# Split into validation and training data\n",
    "train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)\n",
    "\n",
    "# Define a random forest model\n",
    "rf_model = RandomForestRegressor(random_state=1)\n",
    "rf_model.fit(train_X, train_y)\n",
    "rf_val_predictions = rf_model.predict(val_X)\n",
    "rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)\n",
    "\n",
    "print(\"Validation MAE for Random Forest Model: {:,.0f}\".format(rf_val_mae))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3521de4",
   "metadata": {
    "papermill": {
     "duration": 0.006552,
     "end_time": "2023-09-01T09:39:55.943455",
     "exception": false,
     "start_time": "2023-09-01T09:39:55.936903",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Train a model for the competition\n",
    "\n",
    "The code cell above trains a Random Forest model on **`train_X`** and **`train_y`**.  \n",
    "\n",
    "Use the code cell below to build a Random Forest model and train it on all of **`X`** and **`y`**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8010f081",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:39:55.958889Z",
     "iopub.status.busy": "2023-09-01T09:39:55.958437Z",
     "iopub.status.idle": "2023-09-01T09:39:55.966813Z",
     "shell.execute_reply": "2023-09-01T09:39:55.965625Z"
    },
    "papermill": {
     "duration": 0.018743,
     "end_time": "2023-09-01T09:39:55.969045",
     "exception": false,
     "start_time": "2023-09-01T09:39:55.950302",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/markdown": [],
      "text/plain": [
       "<learntools.core.constants.PlaceholderValue at 0x7a73db2785b0>"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# To improve accuracy, create a new Random Forest model which you will train on all training data\n",
    "rf_model_on_full_data = ____\n",
    "\n",
    "# fit rf_model_on_full_data on all data from the training data\n",
    "____"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e28614af",
   "metadata": {
    "papermill": {
     "duration": 0.00666,
     "end_time": "2023-09-01T09:39:55.982935",
     "exception": false,
     "start_time": "2023-09-01T09:39:55.976275",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Now, read the file of \"test\" data, and apply your model to make predictions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "0ccf53a1",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:39:55.998987Z",
     "iopub.status.busy": "2023-09-01T09:39:55.998282Z",
     "iopub.status.idle": "2023-09-01T09:39:56.736070Z",
     "shell.execute_reply": "2023-09-01T09:39:56.735082Z"
    },
    "papermill": {
     "duration": 0.749076,
     "end_time": "2023-09-01T09:39:56.738820",
     "exception": false,
     "start_time": "2023-09-01T09:39:55.989744",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# path to file you will use for predictions\n",
    "test_data_path = '../input/test.csv'\n",
    "\n",
    "# read test data file using pandas\n",
    "test_data = ____\n",
    "\n",
    "# create test_X which comes from test_data but includes only the columns you used for prediction.\n",
    "# The list of columns is stored in a variable called features\n",
    "test_X = ____\n",
    "\n",
    "\n",
    "\n",
    "# In previous code cell\n",
    "rf_model_on_full_data = RandomForestRegressor()\n",
    "rf_model_on_full_data.fit(X, y)\n",
    "\n",
    "# Then in last code cell\n",
    "test_data_path = '../input/test.csv'\n",
    "test_data = pd.read_csv(test_data_path)\n",
    "test_X = test_data[features]\n",
    "test_preds = rf_model_on_full_data.predict(test_X)\n",
    "\n",
    "\n",
    "output = pd.DataFrame({'Id': test_data.Id,\n",
    "                       'SalePrice': test_preds})\n",
    "output.to_csv('submission.csv', index=False)\n",
    "# make predictions which we will submit. \n",
    "test_preds = ____"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0aa02c91",
   "metadata": {
    "papermill": {
     "duration": 0.006948,
     "end_time": "2023-09-01T09:39:56.753147",
     "exception": false,
     "start_time": "2023-09-01T09:39:56.746199",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Before submitting, run a check to make sure your `test_preds` have the right format."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f146b46f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:39:56.769310Z",
     "iopub.status.busy": "2023-09-01T09:39:56.768879Z",
     "iopub.status.idle": "2023-09-01T09:39:56.782876Z",
     "shell.execute_reply": "2023-09-01T09:39:56.781829Z"
    },
    "papermill": {
     "duration": 0.025037,
     "end_time": "2023-09-01T09:39:56.785194",
     "exception": false,
     "start_time": "2023-09-01T09:39:56.760157",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"outcomeType\": 4, \"interactionType\": 1, \"questionType\": 2, \"questionId\": \"1_CheckSubmittablePreds\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\"}}, \"*\")"
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
       "<span style=\"color:#ccaa33\">Check:</span> When you've updated the starter code, `check()` will tell you whether your code is correct. You need to update the code that creates variable `test_preds`"
      ],
      "text/plain": [
       "Check: When you've updated the starter code, `check()` will tell you whether your code is correct. You need to update the code that creates variable `test_preds`"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "parent.postMessage({\"jupyterEvent\": \"custom.exercise_interaction\", \"data\": {\"interactionType\": 3, \"questionType\": 2, \"questionId\": \"1_CheckSubmittablePreds\", \"learnToolsVersion\": \"0.3.4\", \"valueTowardsCompletion\": 0.0, \"failureMessage\": \"\", \"exceptionClass\": \"\", \"trace\": \"\", \"outcomeType\": 4}}, \"*\")"
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
       "\n",
       "# In previous code cell\n",
       "rf_model_on_full_data = RandomForestRegressor()\n",
       "rf_model_on_full_data.fit(X, y)\n",
       "\n",
       "# Then in last code cell\n",
       "test_data_path = '../input/test.csv'\n",
       "test_data = pd.read_csv(test_data_path)\n",
       "test_X = test_data[features]\n",
       "test_preds = rf_model_on_full_data.predict(test_X)\n",
       "\n",
       "\n",
       "output = pd.DataFrame({'Id': test_data.Id,\n",
       "                       'SalePrice': test_preds})\n",
       "output.to_csv('submission.csv', index=False)\n",
       "\n",
       "```"
      ],
      "text/plain": [
       "Solution: \n",
       "```python\n",
       "\n",
       "# In previous code cell\n",
       "rf_model_on_full_data = RandomForestRegressor()\n",
       "rf_model_on_full_data.fit(X, y)\n",
       "\n",
       "# Then in last code cell\n",
       "test_data_path = '../input/test.csv'\n",
       "test_data = pd.read_csv(test_data_path)\n",
       "test_X = test_data[features]\n",
       "test_preds = rf_model_on_full_data.predict(test_X)\n",
       "\n",
       "\n",
       "output = pd.DataFrame({'Id': test_data.Id,\n",
       "                       'SalePrice': test_preds})\n",
       "output.to_csv('submission.csv', index=False)\n",
       "\n",
       "```"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Check your answer (To get credit for completing the exercise, you must get a \"Correct\" result!)\n",
    "step_1.check()\n",
    "step_1.solution()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0bb4dbdf",
   "metadata": {
    "papermill": {
     "duration": 0.007802,
     "end_time": "2023-09-01T09:39:56.800954",
     "exception": false,
     "start_time": "2023-09-01T09:39:56.793152",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Generate a submission\n",
    "\n",
    "Run the code cell below to generate a CSV file with your predictions that you can use to submit to the competition."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "30b2cd69",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:39:56.819109Z",
     "iopub.status.busy": "2023-09-01T09:39:56.818578Z",
     "iopub.status.idle": "2023-09-01T09:39:56.835666Z",
     "shell.execute_reply": "2023-09-01T09:39:56.834541Z"
    },
    "papermill": {
     "duration": 0.029557,
     "end_time": "2023-09-01T09:39:56.838574",
     "exception": false,
     "start_time": "2023-09-01T09:39:56.809017",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "# Run the code to save predictions in the format used for competition scoring\n",
    "\n",
    "output = pd.DataFrame({'Id': test_data.Id,\n",
    "                       'SalePrice': test_preds})\n",
    "output.to_csv('submission.csv', index=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cd724a16",
   "metadata": {
    "papermill": {
     "duration": 0.007541,
     "end_time": "2023-09-01T09:39:56.853991",
     "exception": false,
     "start_time": "2023-09-01T09:39:56.846450",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "# Submit to the competition\n",
    "\n",
    "To test your results, you'll need to join the competition (if you haven't already).  So open a new window by clicking on **[this link](https://www.kaggle.com/c/home-data-for-ml-course)**.  Then click on the **Join Competition** button.\n",
    "\n",
    "![join competition image](https://storage.googleapis.com/kaggle-media/learn/images/axBzctl.png)\n",
    "\n",
    "Next, follow the instructions below:\n",
    "1. Begin by clicking on the **Save Version** button in the top right corner of the window.  This will generate a pop-up window.  \n",
    "2. Ensure that the **Save and Run All** option is selected, and then click on the **Save** button.\n",
    "3. This generates a window in the bottom left corner of the notebook.  After it has finished running, click on the number to the right of the **Save Version** button.  This pulls up a list of versions on the right of the screen.  Click on the ellipsis **(...)** to the right of the most recent version, and select **Open in Viewer**.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.\n",
    "4. Click on the **Data** tab near the top of the screen.  Then, click on the file you would like to submit, and click on the **Submit** button to submit your results to the leaderboard.\n",
    "\n",
    "You have now successfully submitted to the competition!\n",
    "\n",
    "If you want to keep working to improve your performance, select the **Edit** button in the top right of the screen. Then you can change your code and repeat the process. There's a lot of room to improve, and you will climb up the leaderboard as you work.\n",
    "\n",
    "\n",
    "# Continue Your Progress\n",
    "There are many ways to improve your model, and **experimenting is a great way to learn at this point.**\n",
    "\n",
    "The best way to improve your model is to add features.  To add more features to the data, revisit the first code cell, and change this line of code to include more column names:\n",
    "```python\n",
    "features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']\n",
    "```\n",
    "\n",
    "Some features will cause errors because of issues like missing values or non-numeric data types.  Here is a complete list of potential columns that you might like to use, and that won't throw errors:\n",
    "- 'MSSubClass'\n",
    "- 'LotArea'\n",
    "- 'OverallQual' \n",
    "- 'OverallCond' \n",
    "- 'YearBuilt'\n",
    "- 'YearRemodAdd' \n",
    "- '1stFlrSF'\n",
    "- '2ndFlrSF' \n",
    "- 'LowQualFinSF' \n",
    "- 'GrLivArea'\n",
    "- 'FullBath'\n",
    "- 'HalfBath'\n",
    "- 'BedroomAbvGr' \n",
    "- 'KitchenAbvGr' \n",
    "- 'TotRmsAbvGrd' \n",
    "- 'Fireplaces' \n",
    "- 'WoodDeckSF' \n",
    "- 'OpenPorchSF'\n",
    "- 'EnclosedPorch' \n",
    "- '3SsnPorch' \n",
    "- 'ScreenPorch' \n",
    "- 'PoolArea' \n",
    "- 'MiscVal' \n",
    "- 'MoSold' \n",
    "- 'YrSold'\n",
    "\n",
    "Look at the list of columns and think about what might affect home prices.  To learn more about each of these features, take a look at the data description on the **[competition page](https://www.kaggle.com/c/home-data-for-ml-course/data)**.\n",
    "\n",
    "After updating the code cell above that defines the features, re-run all of the code cells to evaluate the model and generate a new submission file.  \n",
    "\n",
    "\n",
    "# What's next?\n",
    "\n",
    "As mentioned above, some of the features will throw an error if you try to use them to train your model.  The **[Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning)** course will teach you how to handle these types of features. You will also learn to use **xgboost**, a technique giving even better accuracy than Random Forest.\n",
    "\n",
    "The **[Pandas](https://kaggle.com/Learn/Pandas)** course will give you the data manipulation skills to quickly go from conceptual idea to implementation in your data science projects. \n",
    "\n",
    "You are also ready for the **[Deep Learning](https://kaggle.com/Learn/intro-to-Deep-Learning)** course, where you will build models with better-than-human level performance at computer vision tasks."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "681915b7",
   "metadata": {
    "papermill": {
     "duration": 0.007624,
     "end_time": "2023-09-01T09:39:56.869431",
     "exception": false,
     "start_time": "2023-09-01T09:39:56.861807",
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
   "duration": 8.375632,
   "end_time": "2023-09-01T09:39:57.500089",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-09-01T09:39:49.124457",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
