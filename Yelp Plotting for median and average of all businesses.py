import json
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from scipy.stats.stats import pearsonr
reviewData = open("yelp_academic_dataset_review.json", "r").readlines()

businessInfo = {}    # Make an empty dictionary
for d in reviewData[0:500000]:
  parsed = json.loads(d)
  business = parsed["business_id"]
  score = parsed["stars"]
  text = parsed["text"]
  tb = TextBlob(text)
  polarity = tb.sentiment.polarity
  businessData = businessInfo.get(business, [])  # Get the list of scores and polarities out of businessInfo. If there's no entry for that business yet, get an empty list.
  businessData.append((score, polarity,))         # Append a tuple to businessData. businessData is always a list of tuples where the first value is the score and the second value is the polarity.
  businessInfo[business] = businessData

for key, value in businessInfo.items():
    print (key, value)

scores = []
polarities = []
allMedianScores = []
allMedianPolarities = []
allMeanScores = []
allMeanPolarities = []
for business, scoreAndPolarity in businessInfo.iteritems():      # Iterates over all the items in the dictionary using the form "key, value".
  for x in range(len(scoreAndPolarity)): # creates a separate list for each business of the separate scores and polarities
  	scores.append((scoreAndPolarity[x])[0])
  	polarities.append((scoreAndPolarity[x])[1])
  medianScore=np.median(scores)
  medianPolarity=np.median(polarities) 
  allMedianScores.append(medianScore)
  allMedianPolarities.append(medianPolarity)
  meanScore=np.mean(scores)
  meanPolarity=np.mean(polarities)
  allMeanScores.append(meanScore)
  allMeanPolarities.append(meanPolarity)

x=np.array(allMedianPolarities)
y=np.array(allMedianScores)
correlation = pearsonr(allMedianPolarities, allMedianScores)
m, b = np.polyfit(x, y, 1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, 'bo')
ax.plot(x, m*x + b, 'r-')
ax.set_title("All Businesses Median Values") 
ax.annotate("Slope="+str(round(m,4)), xy=(1, 0.1), xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='center')
ax.annotate("Pearson correlation="+str(round(correlation[0],4)), xy=(1, 0.05), xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='center')
ax.set_xlabel('median text sentiment polarity')
ax.set_ylabel('median star rating')
ax.set_ylim([0,5])
ax.set_xlim([-1,1])
#  plt.show()
filename = "allBusinessmedians.png"
fig.savefig(filename)
plt.close()

x=np.array(allMeanPolarities)
y=np.array(allMeanScores)
correlation = pearsonr(allMeanPolarities, allMeanScores)
m, b = np.polyfit(x, y, 1)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, 'bo')
ax.plot(x, m*x + b, 'r-')
ax.set_title("All Businesses Mean Values") 
ax.annotate("Slope="+str(round(m,4)), xy=(1, 0.1), xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='center')
ax.annotate("Pearson correlation="+str(round(correlation[0],4)), xy=(1, 0.05), xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='center')
ax.set_xlabel('mean text sentiment polarity')
ax.set_ylabel('mean star rating')
ax.set_ylim([0,5])
ax.set_xlim([-1,1])
#  plt.show()
filename = "allBusinessmeans.png"
fig.savefig(filename)
plt.close()
