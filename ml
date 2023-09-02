{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "4211d023",
   "metadata": {
    "papermill": {
     "duration": 0.005679,
     "end_time": "2023-09-02T01:37:30.683494",
     "exception": false,
     "start_time": "2023-09-02T01:37:30.677815",
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
   "id": "6716eca4",
   "metadata": {
    "papermill": {
     "duration": 0.004707,
     "end_time": "2023-09-02T01:37:30.693417",
     "exception": false,
     "start_time": "2023-09-02T01:37:30.688710",
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
   "id": "c43675ea",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-02T01:37:30.708613Z",
     "iopub.status.busy": "2023-09-02T01:37:30.708260Z",
     "iopub.status.idle": "2023-09-02T01:37:31.117019Z",
     "shell.execute_reply": "2023-09-02T01:37:31.115942Z"
    },
    "papermill": {
     "duration": 0.421051,
     "end_time": "2023-09-02T01:37:31.119823",
     "exception": false,
     "start_time": "2023-09-02T01:37:30.698772",
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
   "id": "99a91b5f",
   "metadata": {
    "papermill": {
     "duration": 0.005033,
     "end_time": "2023-09-02T01:37:31.132307",
     "exception": false,
     "start_time": "2023-09-02T01:37:31.127274",
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
   "id": "0d3e4c6e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-02T01:37:31.144615Z",
     "iopub.status.busy": "2023-09-02T01:37:31.143479Z",
     "iopub.status.idle": "2023-09-02T01:37:33.435986Z",
     "shell.execute_reply": "2023-09-02T01:37:33.434662Z"
    },
    "papermill": {
     "duration": 2.300943,
     "end_time": "2023-09-02T01:37:33.438290",
     "exception": false,
     "start_time": "2023-09-02T01:37:31.137347",
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
   "id": "942ffac4",
   "metadata": {
    "papermill": {
     "duration": 0.004926,
     "end_time": "2023-09-02T01:37:33.449134",
     "exception": false,
     "start_time": "2023-09-02T01:37:33.444208",
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
   "id": "8223a9a9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-02T01:37:33.461657Z",
     "iopub.status.busy": "2023-09-02T01:37:33.460754Z",
     "iopub.status.idle": "2023-09-02T01:37:34.067747Z",
     "shell.execute_reply": "2023-09-02T01:37:34.066729Z"
    },
    "papermill": {
     "duration": 0.615561,
     "end_time": "2023-09-02T01:37:34.069956",
     "exception": false,
     "start_time": "2023-09-02T01:37:33.454395",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>RandomForestRegressor()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestRegressor</label><div class=\"sk-toggleable__content\"><pre>RandomForestRegressor()</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "RandomForestRegressor()"
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
    "rf_model_on_full_data = RandomForestRegressor()\n",
    "rf_model_on_full_data.fit(X, y)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f05044a4",
   "metadata": {
    "papermill": {
     "duration": 0.004907,
     "end_time": "2023-09-02T01:37:34.080643",
     "exception": false,
     "start_time": "2023-09-02T01:37:34.075736",
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
   "id": "54545a8c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-02T01:37:34.092592Z",
     "iopub.status.busy": "2023-09-02T01:37:34.092207Z",
     "iopub.status.idle": "2023-09-02T01:37:34.802253Z",
     "shell.execute_reply": "2023-09-02T01:37:34.800962Z"
    },
    "papermill": {
     "duration": 0.719139,
     "end_time": "2023-09-02T01:37:34.804899",
     "exception": false,
     "start_time": "2023-09-02T01:37:34.085760",
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
    "\n",
    "\n",
    "# Then in last code cell\n",
    "test_data_path = '../input/test.csv'\n",
    "test_data = pd.read_csv(test_data_path)\n",
    "test_X = test_data[features]\n",
    "test_preds = rf_model_on_full_data.predict(test_X)\n",
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
   "id": "eed3df4a",
   "metadata": {
    "papermill": {
     "duration": 0.005108,
     "end_time": "2023-09-02T01:37:34.815649",
     "exception": false,
     "start_time": "2023-09-02T01:37:34.810541",
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
   "id": "261bf43d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-02T01:37:34.828163Z",
     "iopub.status.busy": "2023-09-02T01:37:34.827727Z",
     "iopub.status.idle": "2023-09-02T01:37:34.839627Z",
     "shell.execute_reply": "2023-09-02T01:37:34.838921Z"
    },
    "papermill": {
     "duration": 0.020457,
     "end_time": "2023-09-02T01:37:34.841587",
     "exception": false,
     "start_time": "2023-09-02T01:37:34.821130",
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
   "id": "760709ce",
   "metadata": {
    "papermill": {
     "duration": 0.005634,
     "end_time": "2023-09-02T01:37:34.853084",
     "exception": false,
     "start_time": "2023-09-02T01:37:34.847450",
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
   "id": "2e7bb6b7",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2023-09-02T01:37:34.866742Z",
     "iopub.status.busy": "2023-09-02T01:37:34.866153Z",
     "iopub.status.idle": "2023-09-02T01:37:34.878360Z",
     "shell.execute_reply": "2023-09-02T01:37:34.877190Z"
    },
    "papermill": {
     "duration": 0.021843,
     "end_time": "2023-09-02T01:37:34.880807",
     "exception": false,
     "start_time": "2023-09-02T01:37:34.858964",
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
   "id": "832cc9a1",
   "metadata": {
    "papermill": {
     "duration": 0.005554,
     "end_time": "2023-09-02T01:37:34.892513",
     "exception": false,
     "start_time": "2023-09-02T01:37:34.886959",
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
   "id": "cf4319f2",
   "metadata": {
    "papermill": {
     "duration": 0.005932,
     "end_time": "2023-09-02T01:37:34.904794",
     "exception": false,
     "start_time": "2023-09-02T01:37:34.898862",
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
   "duration": 8.147157,
   "end_time": "2023-09-02T01:37:35.531327",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2023-09-02T01:37:27.384170",
   "version": "2.4.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
