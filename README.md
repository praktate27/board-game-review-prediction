# board-game-review-prediction

# Python libraries used :
Scikit-learn, Pandas and Matplotlib

The dataset is from BoardGameGeek, and contains data on 80000 board games. It is scraped into csv format by Sean Beck.

* Description of data points in Data Set
  --> name : name of the board game.
  --> playingtime : the playing time (given by the manufacturer).
  --> minplaytime : the minimum playing time (given by the manufacturer).
  --> maxplaytime : the maximum playing time (given by the manufacturer).
  --> minage : the minimum recommended age to play.
  --> users_rated : the number of users who rated the game.
  --> average_rating : the average rating given to the game by users. (0-10)
  --> total_weights : Number of weights given by users. Weight is a subjective measure that is made up by BoardGameGeek. It's                        how "deep" or involved a game is.
  --> average_weight : the average of all the subjective weights (0-5).

# Steps:

1. Read the CSV using panda read_csv()

2. A human would give average rating to a new, unreleased, board game. This is stored in the "average_rating" column, which is the average of all the user ratings for a board game. Predicting this column could be useful to board game manufacturers who are thinking of what kind of game to make next, for instance.
After plotting a histogram of column "average_rating", we can see that there are quite few games with a '0' ratings. There's a fairly normal distribution of ratings, with some right skew, and a mean rating around '6' (if zero is removed).

3. In order to find difference between '0' avg reviews and non-zero avg reviews, we will print data of respective and output will be as follow:
* id  -->                           318
* type -->                    boardgame
* name  -->                  Looney Leo
* users_rated  -->                    0
* average_rating  -->                 0
* bayes_average_rating -->            0

Name: 13048, dtype: object

* id   -->                               12333
* type  -->                          boardgame
* name  -->                  Twilight Struggle
* users_rated -->                        20113
* average_rating -->                   8.33774
* bayes_average_rating -->             8.22186

Name: 0, dtype: object

This implies that games with rating '0' has users_rated as '0' which is not case with avg_ratings above '0' reviews;
Hence we need to filter out data with no user ratings and having missing values.

4. Now we will be using K-means algorithm for creating clusters of data. Clustering enables you to find patterns within your data easily by grouping similar rows, together. Scikit-learn has an excellent implementation of k-means clustering that we can use. To use k-means clustering, we need to decide two parameters, n_clusters (defining how many clusters of game we want) and random_state (a random seed we set in order to reproduce our results later).
As most of ML algorithms does not use text data, we will removew type and name columns as they are text data and not useful for creating clusters.
As it is impossible to visualize things in more than 3D, we will reduce the dimensionality of our data by using PCA (Principale Component Analysis). PCA takes multiple columns, and turns them into fewer columns while trying to preserve the unique information in each column. To simplify, say we have two columns, total_owners, and total_traders. There is some correlation between these two columns, and some overlapping information. PCA will compress this information into one column with new numbers while trying not to lose any information.

5. Using Pandas's corr(), we can figure out correlation between columns to predict average_rating. 


* id    -->                      0.304201 
* yearpublished   -->            0.108461  
* minplayers         -->        -0.032701  
* maxplayers            -->     -0.008335  
* playingtime      -->           0.048994  
* minplaytime      -->           0.043985  
* maxplaytime      -->           0.048994  
* minage           -->           0.210049  
* users_rated      -->           0.112564  
* average_rating   -->           1.000000  
* bayes_average_rating -->       0.231563  
* total_owners         -->       0.137478  
* total_traders        -->       0.119452  
* total_wanters         -->      0.196566  
* total_wishers         -->      0.171375  
* total_comments        -->      0.123714  
* total_weights           -->    0.109691  
* average_weight       -->       0.351081  

Name: average_rating, dtype: float64 

the average_weight and id columns correlate best to rating. ids are presumably assigned when the game is added to the database, so this likely indicates that games created later score higher in the ratings. average_weight indicates the "depth" or complexity of a game, so it may be that more complex games are reviewed better.

6. Before we get started predicting, let's only select the columns that are relevant when training our algorithm. We'll want to remove certain columns that aren't numeric. The bayes_average_rating column appears to be derived from average_rating in some way, so we will be removing it.

7. In order to prevent overfitting, we'll train our algorithm on a set consisting of 80% of the data, and test it on another set consisting of 20% of the data.

8. Linear regression is a powerful and commonly used machine learning algorithm. It predicts the target variable using linear combinations of the predictor variables. It only works well when the predictor variables and the target variable are linearly correlated. We are using Scikit-learn for implementation.

9. For predicting errors in our model, we will be using mean squarred to Compute error between our test predictions and the actual values. In mean square error, we subtract each predicted value from the actual value, square the differences, and add them together. Then we divide the result by the total number of predicted values.

10. Another model tries is Random forest whose implementaion Scikit-learn has. The random forest algorithm can find nonlinearities in data that a linear regression wouldn't be able to pick up on. 

For example, that if the minage of a game, is less than 5, the rating is low, if it's 5-10, it's high, and if it is between 10-15, it is low. A linear regression algorithm wouldn't be able to pick up on this because there isn't a linear relationship between the predictor and the target. Predictions made with a random forest usually have less error than predictions made by a linear regression.

