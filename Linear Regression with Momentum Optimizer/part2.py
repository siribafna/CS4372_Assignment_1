{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 13,
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
    "df = pd.read_csv('https://raw.githubusercontent.com/siribafna/CS4372_Assignment_1/master/autos.csv')\n",
    "print(df.head())\n",
    "print('\\nDimensions of data frame:', df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.replace('?', np.nan) # replacing nulls"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
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
     "execution_count": 15,
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
   "execution_count": 16,
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
   "execution_count": 17,
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
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop(columns=['symboling', 'normalized_losses'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
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
   "execution_count": 20,
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
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "x = (X_train - X_train.mean()) / X_train.std()\n",
    "x = np.c_[np.ones(X_train.shape[0]), x] \n",
    "x1 = (X_test - X_test.mean()) / X_test.std()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "intercept: 53.84094488188977\n",
      "coefficients: [0.         0.76742225]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# train the algorithm\n",
    "from sklearn.linear_model import LinearRegression\n",
    "linreg = LinearRegression()\n",
    "linreg.fit(x, y_train)\n",
    "print('intercept:', linreg.intercept_)\n",
    "print('coefficients:', linreg.coef_) # out put produced similar to "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = linreg.predict(x) "
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
