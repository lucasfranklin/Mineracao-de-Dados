pacman::p_load(twitteR)
pacman::p_load(ROAuth)

consumer_key = "PnaFl3Mg9aRNFT1hDTU4wq43j" 
consumer_secret = "fNU7yDnBAFoQjYLUh6xu0mJwftzzM2Xn6mizbwSfYJttKV6nGj"
access_token = "1036744604576636929-BN71lx8ivqoli0f2P2dGOu6gqrmsIY"
access_secret = "sIkPZIrBHHwr4PbCIbd8IbT5ynTVDAoNIlywpXTtVvw5l" 
setup_twitter_oauth(consumer_key, consumer_secret, access_token, access_secret)
 
##### Retrieve Tweets #####

tweets <- userTimeline("jairbolsonaro", n = 3200) # Twitter account and N as Number of Tweets

(n.tweet <- length(tweets)) # Number of Tweets

tweets.df <- twListToDF(tweets) # Converts to data frame

tweets.df[211, c("id", "created", "screenName", "replyToSN", "favoriteCount", "retweetCount", "longitude", "latitude", "text")] # Get data from tweet 190

writeLines(strwrap(tweets.df$text[211], 60)) #Format Tweet Text


## Option 2: download @RDataMining tweets from RDataMining.com
#url <- "http://www.rdatamining.com/data/RDataMining-Tweets-20160212.rds"
#download.file(url, destfile = "./data/RDataMining-Tweets-20160212.rds")
## load tweets into R
#tweets <- readRDS("./data/RDataMining-Tweets-20160212.rds")


#### Text Cleaning ####

pacman::p_load(tm)
# build a corpus, and specify the source to be character vectors
myCorpus <- Corpus(VectorSource(tweets.df$text))
# convert to lower case
myCorpus <- tm_map(myCorpus, content_transformer(tolower))
# remove URLs
removeURL <- function(x) gsub("http[^[:space:]]*", "", x)
myCorpus <- tm_map(myCorpus, content_transformer(removeURL))
# remove anything other than English letters or space
removeNumPunct <- function(x) gsub("[^[:alpha:][:space:]]*", "", x)
myCorpus <- tm_map(myCorpus, content_transformer(removeNumPunct))
# remove stopwords
myStopwords <- c(setdiff(stopwords('portuguese'), c("r", "big")),
                 "use", "see", "used", "via", "amp")
myCorpus <- tm_map(myCorpus, removeWords, myStopwords)
# remove extra whitespace
myCorpus <- tm_map(myCorpus, stripWhitespace)
# keep a copy for stem completion later
myCorpusCopy <- myCorpus

# remove stopwords


#### Stemming and Stem Completion ####
pacman::p_load(SnowballC)
myCorpus <- tm_map(myCorpus, stemDocument) # stem words
writeLines(strwrap(myCorpus[[211]]$content, 60))

stemCompletion2 <- function(x, dictionary) {
  x <- unlist(strsplit(as.character(x), " "))
  x <- x[x != ""]
  x <- stemCompletion(x, dictionary=dictionary)
  x <- paste(x, sep="", collapse=" ")
  PlainTextDocument(stripWhitespace(x))
}

myCorpus <- lapply(myCorpus, stemCompletion2, dictionary=myCorpusCopy)
myCorpus <- Corpus(VectorSource(myCorpus))
writeLines(strwrap(myCorpus[[211]]$content, 60))

#### Issues in Stem Completion: \Miner" vs \Mining" ####
# count word frequence
wordFreq <- function(corpus, word) {
  results <- lapply(corpus, 
                    function(x) { grep(as.character(x), pattern=paste0("nn<",word)) }
)
  sum(unlist(results))
}
n.miner <- wordFreq(myCorpus, "foto")
n.mining <- wordFreq(myCorpus, "bolsonaro")
cat(n.miner, n.mining)

# replace oldword with newword
replaceWord <- function(corpus, oldword, newword) {
tm_map(corpus, content_transformer(gsub),
       pattern=oldword, replacement=newword)
}
myCorpus <- replaceWord(myCorpus, "miner", "mining")
myCorpus <- replaceWord(myCorpus, "universidad", "university")
myCorpus <- replaceWord(myCorpus, "scienc", "science")

#### Build Term Document Matrix ####
tdm <- TermDocumentMatrix(myCorpus,
                          control = list(wordLengths = c(1, Inf)))
tdm

idx <- which(dimnames(tdm)$Terms %in% c("campanha", "presidente", "brasil"))
as.matrix(tdm[idx, 21:30])

(freq.terms <- findFreqTerms(tdm, lowfreq = 20))

term.freq <- rowSums(as.matrix(tdm))
term.freq <- subset(term.freq, term.freq >= 20)
df <- data.frame(term = names(term.freq), freq = term.freq)

pacman::p_load(ggplot2)

ggplot(df, aes(x=term, y=freq)) + geom_bar(stat="identity") +
  xlab("Terms") + ylab("Count") + coord_flip() +
  theme(axis.text=element_text(size=7))


#### Wordcloud ####
pacman::p_load(RColorBrewer)
m <- as.matrix(tdm)
# calculate the frequency of words and sort it by frequency
word.freq <- sort(rowSums(m), decreasing = T)
# colors
pal <- brewer.pal(9, "BuGn")[-(1:4)]

pacman::p_load(wordcloud)
wordcloud(words = names(word.freq), freq = word.freq, min.freq = 3,
          random.order = F, colors = pal)

#### Associations ####
findAssocs(tdm, "r", 0.2)
findAssocs(tdm, "data", 0.2)

#### Network of Terms ####
pacman::p_load(graph)
pacman::p_load(Rgraphviz)
plot(tdm, term = freq.terms, corThreshold = 0.1, weighting = T)

#### Topic Modelling ####
dtm <- as.DocumentTermMatrix(tdm)
pacman::p_load(topicmodels)
lda <- LDA(dtm, k = 8) # find 8 topics
term <- terms(lda, 7) # first 7 terms of every topic
(term <- apply(term, MARGIN = 2, paste, collapse = ", "))
#ERROR#
pacman::p_load(data.table)

topics <- topics(lda) # 1st topic identified for every document (tweet)
topics <- data.frame(date=as.IDate(tweets.df$created), topic=topics)
ggplot(topics, aes(date, fill = term[topic])) +
  geom_density(position = "stack")

#### Sentiment Analysis ####
# install package sentiment140
require(devtools)
install_github('sentiment140', 'okugami79')
pacman::p_load(sentiment)

# sentiment analysis
sentiments <- sentiment(tweets.df$text)
table(sentiments$polarity)

# sentiment plot
sentiments$score <- 0
sentiments$score[sentiments$polarity == "positive"] <- 1
sentiments$score[sentiments$polarity == "negative"] <- -1
sentiments$date <- as.IDate(tweets.df$created)
result <- aggregate(score ~ date, data = sentiments, sum)
plot(result, type = "l")