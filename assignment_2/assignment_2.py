{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(146, 6) (31, 6) (32, 6) (146,) (31,) (32,)\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/wheat-seeds.csv')\n",
    "dataset = df.values\n",
    "X = dataset[:,0:6]\n",
    "Y = dataset[:,7]\n",
    "\n",
    "from sklearn import preprocessing\n",
    "min_max_scaler = preprocessing.MinMaxScaler()\n",
    "X_scale = min_max_scaler.fit_transform(X)\n",
    "X_scale\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)\n",
    "X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)\n",
    "print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "input_features = X_train\n",
    "target_output = Y_train.reshape(146,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "metadata": {},
   "outputs": [],
   "source": [
    "def sigmoid(x):\n",
    "    return 1 / (1 + np.exp(-x))\n",
    "\n",
    "def sigmoid_deriv(x):\n",
    "    return sigmoid(x) * (1 - sigmoid(x))\n",
    "\n",
    "def tanh(x):\n",
    "    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))\n",
    "\n",
    "def tanh_deriv(x):\n",
    "    return sigmoid(x) * (1 - sigmoid(x))\n",
    "\n",
    "def relu(x):\n",
    "    return np.maximum(0, x)\n",
    "\n",
    "def relu_deriv(x):\n",
    "    if x.any() < 0:\n",
    "        return 0\n",
    "    if x.any() >= 0:\n",
    "        return 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [],
   "source": [
    "def run():\n",
    "    #parameters\n",
    "    weight_hidden = np.random.rand(6,6)\n",
    "    weight_output = np.random.rand(6,6)\n",
    "    lr = .001\n",
    "    momentum_factor = .75 # gradient descent optimizer\n",
    "    activation_function = sigmoid;\n",
    "    af_derivative = sigmoid_deriv;\n",
    "    \n",
    "    for epoch in range(20):\n",
    "\n",
    "        # set up initial variables\n",
    "        hidden_layer_input = np.dot(input_features, weight_hidden)\n",
    "        hidden_layer_output = activation_function(hidden_layer_input)\n",
    "        output_layer_input = np.dot(hidden_layer_output, weight_output)\n",
    "        output_layer_output = activation_function(output_layer_input)\n",
    "\n",
    "        mse = ((1/2 * np.power((output_layer_output - target_output), 2)))\n",
    "        #print(mse)\n",
    "\n",
    "        first_derivative_error = output_layer_output - target_output\n",
    "        out_over_in = af_derivative(output_layer_input)\n",
    "        deriv_in_over_dwo = hidden_layer_output\n",
    "\n",
    "        error_prod_dwo = np.dot(deriv_in_over_dwo.T, first_derivative_error * out_over_in)\n",
    "\n",
    "        deriv_error_in = first_derivative_error * out_over_in\n",
    "        deriv_in_out = weight_output\n",
    "        derror_douth = np.dot(deriv_error_in, deriv_in_out.T)\n",
    "        deriv_out_in = af_derivative(hidden_layer_input)\n",
    "        deriv_input = input_features\n",
    "        error_prod_wh = np.dot(deriv_input.T, deriv_out_in * derror_douth)\n",
    "\n",
    "        weight_hidden = lr * error_prod_wh + (momentum_factor * weight_hidden)\n",
    "        weight_output = lr * error_prod_dwo + (momentum_factor * weight_output)\n",
    "    \n",
    "run();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0.9353317  0.91272659 0.89055088 0.87906644 0.93130549 0.92600475]\n"
     ]
    }
   ],
   "source": [
    "# print (weight_hidden)# Final output layer weight values :\n",
    "# print (weight_output)# Predictions :#Taking inputs :\n",
    "single_point = X_test[1]\n",
    "#1st step :\n",
    "result1 = np.dot(single_point, weight_hidden) \n",
    "#2nd step :\n",
    "result2 = sigmoid(result1)\n",
    "#3rd step :\n",
    "result3 = np.dot(result2,weight_output)\n",
    "#4th step :\n",
    "result4 = sigmoid(result3)\n",
    "print(result4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.91473654 0.88268687 0.86444842 0.85562723 0.90677962 0.90009745]\n",
      " [0.89539282 0.87560081 0.84271455 0.82599766 0.88448836 0.88098142]\n",
      " [0.90934033 0.88955651 0.85747201 0.84370164 0.90156921 0.89668978]\n",
      " [0.89884153 0.86646739 0.84947346 0.8489618  0.89875712 0.88941338]\n",
      " [0.90251045 0.86750558 0.85049356 0.83529208 0.89483525 0.88903555]\n",
      " [0.92168919 0.88404299 0.87192674 0.85564129 0.9028908  0.89623255]]\n"
     ]
    }
   ],
   "source": [
    "# print (weight_hidden)# Final output layer weight values :\n",
    "# print (weight_output)# Predictions :#Taking inputs :\n",
    "single_point = Y_test[1]\n",
    "#1st step :\n",
    "result1 = np.dot(single_point, weight_hidden) \n",
    "#2nd step :\n",
    "result2 = sigmoid(result1)\n",
    "#3rd step :\n",
    "result3 = np.dot(result2,weight_output)\n",
    "#4th step :\n",
    "result4 = sigmoid(result3)\n",
    "print(result4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
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
   "version": "3.8.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
