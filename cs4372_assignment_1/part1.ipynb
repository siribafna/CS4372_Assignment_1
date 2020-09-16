{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   symboling normalized_losses         make fuel_type aspiration num_of_doors  \\\n",
      "0          3                 ?  alfa-romero       gas        std          two   \n",
      "1          3                 ?  alfa-romero       gas        std          two   \n",
      "2          1                 ?  alfa-romero       gas        std          two   \n",
      "3          2               164         audi       gas        std         four   \n",
      "4          2               164         audi       gas        std         four   \n",
      "\n",
      "    body_style drive_wheels engine_location  wheel_base  ...  engine_size  \\\n",
      "0  convertible          rwd           front        88.6  ...          130   \n",
      "1  convertible          rwd           front        88.6  ...          130   \n",
      "2    hatchback          rwd           front        94.5  ...          152   \n",
      "3        sedan          fwd           front        99.8  ...          109   \n",
      "4        sedan          4wd           front        99.4  ...          136   \n",
      "\n",
      "   fuel_system  bore  stroke compression_ratio horsepower  peak_rpm city_mpg  \\\n",
      "0         mpfi  3.47    2.68               9.0        111      5000       21   \n",
      "1         mpfi  3.47    2.68               9.0        111      5000       21   \n",
      "2         mpfi  2.68    3.47               9.0        154      5000       19   \n",
      "3         mpfi  3.19     3.4              10.0        102      5500       24   \n",
      "4         mpfi  3.19     3.4               8.0        115      5500       18   \n",
      "\n",
      "  highway_mpg  price  \n",
      "0          27  13495  \n",
      "1          27  16500  \n",
      "2          26  16500  \n",
      "3          30  13950  \n",
      "4          22  17450  \n",
      "\n",
      "[5 rows x 26 columns]\n",
      "\n",
      "Dimensions of data frame: (205, 26)\n"
     ]
    }
   ],
   "source": [
    "### load the data\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/siribafna/CS4372_Assignment_1/master/autos.csv') # data on the web\n",
    "print(df.head())\n",
    "print('\\nDimensions of data frame:', df.shape) # data exploration functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.replace('?', np.nan) # replace all the nulls "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "symboling             0\n",
       "normalized_losses    41\n",
       "make                  0\n",
       "fuel_type             0\n",
       "aspiration            0\n",
       "num_of_doors          2\n",
       "body_style            0\n",
       "drive_wheels          0\n",
       "engine_location       0\n",
       "wheel_base            0\n",
       "length                0\n",
       "width                 0\n",
       "height                0\n",
       "curb_weight           0\n",
       "engine_type           0\n",
       "num_of_cylinders      0\n",
       "engine_size           0\n",
       "fuel_system           0\n",
       "bore                  4\n",
       "stroke                4\n",
       "compression_ratio     0\n",
       "horsepower            2\n",
       "peak_rpm              2\n",
       "city_mpg              0\n",
       "highway_mpg           0\n",
       "price                 4\n",
       "dtype: int64"
      ]
     },
     "execution_count": 128,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum() # how many nulls there are in each column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Dimensions of data frame: (159, 26)\n"
     ]
    }
   ],
   "source": [
    "df = df.dropna()\n",
    "print('\\nDimensions of data frame:', df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>symboling</th>\n",
       "      <th>normalized_losses</th>\n",
       "      <th>make</th>\n",
       "      <th>fuel_type</th>\n",
       "      <th>aspiration</th>\n",
       "      <th>num_of_doors</th>\n",
       "      <th>body_style</th>\n",
       "      <th>drive_wheels</th>\n",
       "      <th>engine_location</th>\n",
       "      <th>wheel_base</th>\n",
       "      <th>...</th>\n",
       "      <th>engine_size</th>\n",
       "      <th>fuel_system</th>\n",
       "      <th>bore</th>\n",
       "      <th>stroke</th>\n",
       "      <th>compression_ratio</th>\n",
       "      <th>horsepower</th>\n",
       "      <th>peak_rpm</th>\n",
       "      <th>city_mpg</th>\n",
       "      <th>highway_mpg</th>\n",
       "      <th>price</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>164</td>\n",
       "      <td>audi</td>\n",
       "      <td>gas</td>\n",
       "      <td>std</td>\n",
       "      <td>four</td>\n",
       "      <td>sedan</td>\n",
       "      <td>fwd</td>\n",
       "      <td>front</td>\n",
       "      <td>99.8</td>\n",
       "      <td>...</td>\n",
       "      <td>109</td>\n",
       "      <td>mpfi</td>\n",
       "      <td>3.19</td>\n",
       "      <td>3.4</td>\n",
       "      <td>10.0</td>\n",
       "      <td>102</td>\n",
       "      <td>5500</td>\n",
       "      <td>24</td>\n",
       "      <td>30</td>\n",
       "      <td>13950</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2</td>\n",
       "      <td>164</td>\n",
       "      <td>audi</td>\n",
       "      <td>gas</td>\n",
       "      <td>std</td>\n",
       "      <td>four</td>\n",
       "      <td>sedan</td>\n",
       "      <td>4wd</td>\n",
       "      <td>front</td>\n",
       "      <td>99.4</td>\n",
       "      <td>...</td>\n",
       "      <td>136</td>\n",
       "      <td>mpfi</td>\n",
       "      <td>3.19</td>\n",
       "      <td>3.4</td>\n",
       "      <td>8.0</td>\n",
       "      <td>115</td>\n",
       "      <td>5500</td>\n",
       "      <td>18</td>\n",
       "      <td>22</td>\n",
       "      <td>17450</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>1</td>\n",
       "      <td>158</td>\n",
       "      <td>audi</td>\n",
       "      <td>gas</td>\n",
       "      <td>std</td>\n",
       "      <td>four</td>\n",
       "      <td>sedan</td>\n",
       "      <td>fwd</td>\n",
       "      <td>front</td>\n",
       "      <td>105.8</td>\n",
       "      <td>...</td>\n",
       "      <td>136</td>\n",
       "      <td>mpfi</td>\n",
       "      <td>3.19</td>\n",
       "      <td>3.4</td>\n",
       "      <td>8.5</td>\n",
       "      <td>110</td>\n",
       "      <td>5500</td>\n",
       "      <td>19</td>\n",
       "      <td>25</td>\n",
       "      <td>17710</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>1</td>\n",
       "      <td>158</td>\n",
       "      <td>audi</td>\n",
       "      <td>gas</td>\n",
       "      <td>turbo</td>\n",
       "      <td>four</td>\n",
       "      <td>sedan</td>\n",
       "      <td>fwd</td>\n",
       "      <td>front</td>\n",
       "      <td>105.8</td>\n",
       "      <td>...</td>\n",
       "      <td>131</td>\n",
       "      <td>mpfi</td>\n",
       "      <td>3.13</td>\n",
       "      <td>3.4</td>\n",
       "      <td>8.3</td>\n",
       "      <td>140</td>\n",
       "      <td>5500</td>\n",
       "      <td>17</td>\n",
       "      <td>20</td>\n",
       "      <td>23875</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>2</td>\n",
       "      <td>192</td>\n",
       "      <td>bmw</td>\n",
       "      <td>gas</td>\n",
       "      <td>std</td>\n",
       "      <td>two</td>\n",
       "      <td>sedan</td>\n",
       "      <td>rwd</td>\n",
       "      <td>front</td>\n",
       "      <td>101.2</td>\n",
       "      <td>...</td>\n",
       "      <td>108</td>\n",
       "      <td>mpfi</td>\n",
       "      <td>3.5</td>\n",
       "      <td>2.8</td>\n",
       "      <td>8.8</td>\n",
       "      <td>101</td>\n",
       "      <td>5800</td>\n",
       "      <td>23</td>\n",
       "      <td>29</td>\n",
       "      <td>16430</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 26 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "    symboling normalized_losses  make fuel_type aspiration num_of_doors  \\\n",
       "3           2               164  audi       gas        std         four   \n",
       "4           2               164  audi       gas        std         four   \n",
       "6           1               158  audi       gas        std         four   \n",
       "8           1               158  audi       gas      turbo         four   \n",
       "10          2               192   bmw       gas        std          two   \n",
       "\n",
       "   body_style drive_wheels engine_location  wheel_base  ...  engine_size  \\\n",
       "3       sedan          fwd           front        99.8  ...          109   \n",
       "4       sedan          4wd           front        99.4  ...          136   \n",
       "6       sedan          fwd           front       105.8  ...          136   \n",
       "8       sedan          fwd           front       105.8  ...          131   \n",
       "10      sedan          rwd           front       101.2  ...          108   \n",
       "\n",
       "    fuel_system  bore  stroke compression_ratio horsepower  peak_rpm city_mpg  \\\n",
       "3          mpfi  3.19     3.4              10.0        102      5500       24   \n",
       "4          mpfi  3.19     3.4               8.0        115      5500       18   \n",
       "6          mpfi  3.19     3.4               8.5        110      5500       19   \n",
       "8          mpfi  3.13     3.4               8.3        140      5500       17   \n",
       "10         mpfi   3.5     2.8               8.8        101      5800       23   \n",
       "\n",
       "   highway_mpg  price  \n",
       "3           30  13950  \n",
       "4           22  17450  \n",
       "6           25  17710  \n",
       "8           20  23875  \n",
       "10          29  16430  \n",
       "\n",
       "[5 rows x 26 columns]"
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head() # data exploration"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop(columns=['symboling', 'normalized_losses']) # dropping unrelated columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.make = df.make.astype('category').cat.codes\n",
    "df.fuel_type = df.fuel_type.astype('category').cat.codes\n",
    "df.aspiration = df.aspiration.astype('category').cat.codes\n",
    "df.num_of_doors = df.num_of_doors.astype('category').cat.codes\n",
    "df.body_style = df.body_style.astype('category').cat.codes\n",
    "df.drive_wheels = df.drive_wheels.astype('category').cat.codes\n",
    "df.engine_type = df.engine_type.astype('category').cat.codes\n",
    "df.num_of_cylinders = df.num_of_cylinders.astype('category').cat.codes\n",
    "df.fuel_system = df.fuel_system.astype('category').cat.codes\n",
    "df.bore = df.bore.astype('category').cat.codes\n",
    "df.stroke = df.stroke.astype('category').cat.codes\n",
    "df.horsepower = df.horsepower.astype('category').cat.codes\n",
    "df.peak_rpm = df.peak_rpm.astype('category').cat.codes\n",
    "df.price = df.price.astype('category').cat.codes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "train size: (127,)\n",
      "test size: (32,)\n"
     ]
    }
   ],
   "source": [
    "# train test split\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X = df.iloc[:,9]\n",
    "y = df.iloc[:,10]\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)\n",
    "\n",
    "print('train size:', X_train.shape)\n",
    "print('test size:', X_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = (X_train - X_train.mean()) / X_train.std()\n",
    "x = np.c_[np.ones(X_train.shape[0]), x] "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Gradient Descent: 53.84, 0.77\n"
     ]
    }
   ],
   "source": [
    "# gradient descent using momentum optimizer\n",
    "lr = .1\n",
    "iterations = 500\n",
    "num_points = y_train.size \n",
    "np.random.seed(123)\n",
    "theta = np.random.rand(2)\n",
    "gamma = .9\n",
    "\n",
    "def gradient_descent(x, y_train, theta, iterations, lr):\n",
    "    costs = []\n",
    "    thetas = [theta] # prev thetas\n",
    "    momentum = 0;\n",
    "    for i in range(iterations):\n",
    "        pred = np.dot(x, theta)\n",
    "        error = pred - y_train\n",
    "        cost = 1/(2*num_points) * np.dot(error.T, error)\n",
    "        costs.append(cost)\n",
    "        momentum = gamma*momentum + (lr * (1/num_points) * np.dot(x.T, error))\n",
    "        theta = theta - momentum\n",
    "        thetas.append(theta)\n",
    "        log.append(theta)\n",
    "        mse.append(mean_squared_error(y, (theta[0]*X + theta[1]))) \n",
    "    return thetas, costs\n",
    "thetas, costs = gradient_descent(x, y_train, theta, iterations, lr)\n",
    "theta = thetas[-1]\n",
    "\n",
    "print(\"Gradient Descent: {:.2f}, {:.2f}\".format(theta[0], theta[1]))\n",
    "y_preds = theta[0] + theta[1]*x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# visualizing cost function\n",
    "import matplotlib.pyplot as plt\n",
    "plt.title('Cost Function')\n",
    "plt.xlabel('iterations')\n",
    "plt.ylabel('cost')\n",
    "plt.plot(costs)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = df.width.to_numpy()\n",
    "y = df.length.to_numpy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1e9c4fd0>,\n",
       " <matplotlib.lines.Line2D at 0x1dbbf040>]"
      ]
     },
     "execution_count": 97,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x504 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.figure(figsize=(10,7))\n",
    "plt.scatter(X_train,y_train,s=10, alpha=.5, label='Y')\n",
    "plt.plot(X_train, y_preds)"
   ]
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
