{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "d636cf8f",
   "metadata": {
    "papermill": {
     "duration": 0.008405,
     "end_time": "2023-09-01T09:41:58.214418",
     "exception": false,
     "start_time": "2023-09-01T09:41:58.206013",
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
   "id": "a0f76cfa",
   "metadata": {
    "papermill": {
     "duration": 0.006888,
     "end_time": "2023-09-01T09:41:58.229033",
     "exception": false,
     "start_time": "2023-09-01T09:41:58.222145",
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
   "id": "a0b84fd6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:41:58.246499Z",
     "iopub.status.busy": "2023-09-01T09:41:58.246085Z",
     "iopub.status.idle": "2023-09-01T09:41:59.287972Z",
     "shell.execute_reply": "2023-09-01T09:41:59.286299Z"
    },
    "papermill": {
     "duration": 1.055286,
     "end_time": "2023-09-01T09:41:59.291941",
     "exception": false,
     "start_time": "2023-09-01T09:41:58.236655",
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
   "id": "4c1409b3",
   "metadata": {
    "papermill": {
     "duration": 0.009795,
     "end_time": "2023-09-01T09:41:59.310184",
     "exception": false,
     "start_time": "2023-09-01T09:41:59.300389",
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
   "id": "9c054846",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:41:59.331451Z",
     "iopub.status.busy": "2023-09-01T09:41:59.330138Z",
     "iopub.status.idle": "2023-09-01T09:42:02.234427Z",
     "shell.execute_reply": "2023-09-01T09:42:02.233027Z"
    },
    "papermill": {
     "duration": 2.918504,
     "end_time": "2023-09-01T09:42:02.237421",
     "exception": false,
     "start_time": "2023-09-01T09:41:59.318917",
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
   "id": "7b81c567",
   "metadata": {
    "papermill": {
     "duration": 0.006851,
     "end_time": "2023-09-01T09:42:02.251759",
     "exception": false,
     "start_time": "2023-09-01T09:42:02.244908",
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
   "id": "77ee6673",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:42:02.268933Z",
     "iopub.status.busy": "2023-09-01T09:42:02.268401Z",
     "iopub.status.idle": "2023-09-01T09:42:02.278924Z",
     "shell.execute_reply": "2023-09-01T09:42:02.277320Z"
    },
    "papermill": {
     "duration": 0.022948,
     "end_time": "2023-09-01T09:42:02.281811",
     "exception": false,
     "start_time": "2023-09-01T09:42:02.258863",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/markdown": [],
      "text/plain": [
       "<learntools.core.constants.PlaceholderValue at 0x7be5856d45b0>"
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
   "id": "1e243b4a",
   "metadata": {
    "papermill": {
     "duration": 0.007022,
     "end_time": "2023-09-01T09:42:02.296541",
     "exception": false,
     "start_time": "2023-09-01T09:42:02.289519",
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
   "id": "c7a08747",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:42:02.313459Z",
     "iopub.status.busy": "2023-09-01T09:42:02.312893Z",
     "iopub.status.idle": "2023-09-01T09:42:03.095365Z",
     "shell.execute_reply": "2023-09-01T09:42:03.093700Z"
    },
    "papermill": {
     "duration": 0.795139,
     "end_time": "2023-09-01T09:42:03.098976",
     "exception": false,
     "start_time": "2023-09-01T09:42:02.303837",
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
   "id": "dea021ca",
   "metadata": {
    "papermill": {
     "duration": 0.008021,
     "end_time": "2023-09-01T09:42:03.115759",
     "exception": false,
     "start_time": "2023-09-01T09:42:03.107738",
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
   "id": "6b58c416",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:42:03.133800Z",
     "iopub.status.busy": "2023-09-01T09:42:03.133276Z",
     "iopub.status.idle": "2023-09-01T09:42:03.151318Z",
     "shell.execute_reply": "2023-09-01T09:42:03.150115Z"
    },
    "papermill": {
     "duration": 0.030646,
     "end_time": "2023-09-01T09:42:03.154314",
     "exception": false,
     "start_time": "2023-09-01T09:42:03.123668",
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
   "id": "7e4ec695",
   "metadata": {
    "papermill": {
     "duration": 0.007792,
     "end_time": "2023-09-01T09:42:03.170676",
     "exception": false,
     "start_time": "2023-09-01T09:42:03.162884",
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
   "id": "92f62464",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-01T09:42:03.189243Z",
     "iopub.status.busy": "2023-09-01T09:42:03.188667Z",
     "iopub.status.idle": "2023-09-01T09:42:03.208618Z",
     "shell.execute_reply": "2023-09-01T09:42:03.207150Z"
    },
    "papermill": {
     "duration": 0.033479,
     "end_time": "2023-09-01T09:42:03.212113",
     "exception": false,
     "start_time": "2023-09-01T09:42:03.178634",
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
   "id": "efecdfa0",
   "metadata": {
    "papermill": {
     "duration": 0.008965,
     "end_time": "2023-09-01T09:42:03.229677",
     "exception": false,
     "start_time": "2023-09-01T09:42:03.220712",
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
   "id": "2df15782",
   "metadata": {
    "papermill": {
     "duration": 0.008053,
     "end_time": "2023-09-01T09:42:03.246143",
     "exception": false,
     "start_time": "2023-09-01T09:42:03.238090",
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
   "duration": 11.363362,
   "end_time": "2023-09-01T09:42:04.180445",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-09-01T09:41:52.817083",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
