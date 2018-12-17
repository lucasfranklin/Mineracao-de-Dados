
#Load Data
states <- read.csv("~/Data Mining/ClusterData.csv", header = T)
colnames(states)

#Save numerical only
st <- states[, 3:27]
row.names(st) <- states[,2]
colnames(st)

# Sports search data only
sports <- st[,8:11]
head(sports)

# CLUSTERING

# Create distance matrix
d <- dist(st)

# Hierarchical Clustering
c <- hclust(d)
c

#plot
plot(c, main = "Cluster with All Searches and Personality")

#Sports data
d <- dist(sports)
c <- hclust(d)
c
plot(c, main = "Sports Search")
