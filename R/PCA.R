pacman::p_load(psych)
pacman::p_depends(psych)
pacman::p_load(GPArotation)

# Reading CSV
b5 <- read.csv("~/Data Mining/b5.csv", header = T)
colnames(b5)
boxplot(b5)

# Principal Component Analysis
pc0 <- psych::principal(b5, nfactors = 5)
pc0

# Principal Component Analysis with Rotation
pc1 <- psych::principal(b5, nfactors = 5, rotate = "oblimin")
pc1

# Plot PCA
plot(pc1)
