from NewsSentiment import TargetSentimentClassifier
tsc = TargetSentimentClassifier()

# data = [
#     ("I like ", "Peter", " but I don't like Robert."),
#     ("", "Mark Meadows", "'s coverup of Trump’s coup attempt is falling apart."),
# ]
#
# sentiments = tsc.infer(targets=data)
#
# for i, result in enumerate(sentiments):
#     print("Sentiment: ", i, result[0])

sentiment = tsc.infer_from_text("I like " ,"Peter", " but I don't like Robert.")
print(sentiment[0])
