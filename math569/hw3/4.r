## R code for Chapter 4 in The Elements of Statistical Learning (ESL)

## Load R package for the book
library("ElemStatLearn")

## load the training data
data(vowel.train)
str(vowel.train)
# 'data.frame':	528 obs. of  11 variables:
# $ y   : int  1 2 3 4 5 6 7 8 9 10 ...
# $ x.1 : num  -3.64 -3.33 -2.12 -2.29 -2.6 ...
# $ x.2 : num  0.418 0.496 0.894 1.809 1.938 ...
table(vowel.train$y)
#  1  2  3  4  5  6  7  8  9 10 11 
# 48 48 48 48 48 48 48 48 48 48 48 
# y=vowel.train$y
# x=as.matrix(vowel.train[,-1])

## I) Linear Discriminant Analysis, Section 4.3 in ESL
library(MASS)
lda.fit=lda(y ~ ., data=vowel.train)
lda.fit
plot(lda.fit)

## check training errors, Table 4.1, page 107
lda.pred=predict(lda.fit, vowel.train)
names(lda.pred)
lda.class=lda.pred$class
table(lda.class, vowel.train$y)
mean(lda.class!=vowel.train$y) #0.3162879

## check testing errors, Table 4.1, page 107
data(vowel.test)
lda.pred=predict(lda.fit, newdata=vowel.test)
names(lda.pred)
lda.class=lda.pred$class
table(lda.class,vowel.test$y)
mean(lda.class!=vowel.test$y) #0.5562771


## II) Quadratic Discriminant Analysis, Section 4.3, page 110
qda.fit=qda(y ~ ., data=vowel.train)
qda.fit
## check training errors, Table 4.1, page 107
qda.pred=predict(qda.fit, vowel.train)
names(qda.pred)
qda.class=qda.pred$class
table(qda.class, vowel.train$y)
mean(qda.class!=vowel.train$y) #0.01136364
## check testing errors, Table 4.1, page 107 
qda.pred=predict(qda.fit, newdata=vowel.test)
names(qda.pred)
qda.class=qda.pred$class
table(qda.class,vowel.test$y)
mean(qda.class!=vowel.test$y) #0.5281385


## III) find optimal subspaces for LDA, Section 4.3.3, pages 113~115
K <- 11              # 11 vowels or 11 classes
p <- 10              # dimension of input space
n <- dim(vowel.train)[1]                # number of observations
y <- vowel.train$y   # output or response, integer values 1, 2, ..., 11
X <- vowel.train[,2:(p+1)]   # input values

# 11 class centroids in R^10
M <- matrix(0, K, p)                    
for(i in 1:K) M[i,] <- apply(X[y==i,], 2, mean);

# calculate within-class covariance
W <- t(as.matrix(X) - M[y, ])%*%(as.matrix(X) - M[y, ])/n  

# calculate Mstar = M W^{-1/2}
temp <- svd(W)       # singular value decomposition
Wn0p5 <- temp$u %*% diag(1/sqrt(temp$d)) %*% t(temp$v) # W^{-1/2}
Mstar <- M %*% Wn0p5

# calculate covariance of Mstar
Bstar <- var(Mstar)
# alternative way
temp <- Mstar
for(i in 1:10) temp[,i] <- temp[,i]-mean(temp[,i]);
Bstar <- t(temp)%*%temp/10

# eigen-decomposition of Bstar
temp <- eigen(Bstar)
Vstar <- temp$vectors   # columns are v^*_l
V <- Wn0p5 %*% Vstar    # columns are v_l

# discriminant variables
Z <- as.matrix(X) %*% V
for(i in 1:10) Z[,i] <- Z[,i] - mean(Z[,i]);
Z <- -Z
Z[,2] <- -Z[,2]
Z[,10] <- -Z[,10]
Mnew <- matrix(0, K, p)                    
for(i in 1:K) Mnew[i,] <- apply(Z[y==i,], 2, mean);


## plot Figure 4.4 (page 107) and Figure 4.8 (page 115)
# Figure 4.4
par(mfrow=c(1,1))
i1 <- 1
i2 <- 2
plot(Z[,i1], Z[,i2], xlab="Coordinate 1 for Training Data", ylab="Coordinate 2 for Training Data", 
     main="Linear Discriminant Analysis", type="n")
for(i in 1:11) {
  points(Mnew[i,i1], Mnew[i,i2], col=i, type="p", pch="o", cex=2);
  points(Z[y==i,i1], Z[y==i,i2], col=i, type="p", pch=21, cex=1);
}
# Figure 4.8
par(mfrow=c(2,2))
i1 <- 1
i2 <- 3
plot(Z[,i1], Z[,i2], xlab="Coordinate 1", ylab="Coordinate 3", type="n")
for(i in 1:11) {
  points(Mnew[i,i1], Mnew[i,i2], col=i, type="p", pch="o", cex=1.5);
  points(Z[y==i,i1], Z[y==i,i2], col=i, type="p", pch=21, cex=0.7);
}
i1 <- 2
i2 <- 3
plot(Z[,i1], Z[,i2], xlab="Coordinate 2", ylab="Coordinate 3", type="n")
for(i in 1:11) {
  points(Mnew[i,i1], Mnew[i,i2], col=i, type="p", pch="o", cex=1.5);
  points(Z[y==i,i1], Z[y==i,i2], col=i, type="p", pch=21, cex=0.7);
}
i1 <- 1
i2 <- 7
plot(Z[,i1], Z[,i2], xlab="Coordinate 1", ylab="Coordinate 7", type="n")
for(i in 1:11) {
  points(Mnew[i,i1], Mnew[i,i2], col=i, type="p", pch="o", cex=1.5);
  points(Z[y==i,i1], Z[y==i,i2], col=i, type="p", pch=21, cex=0.7);
}
i1 <- 9
i2 <- 10
plot(Z[,i1], Z[,i2], xlab="Coordinate 9", ylab="Coordinate 10", type="n")
for(i in 1:11) {
  points(Mnew[i,i1], Mnew[i,i2], col=i, type="p", pch="o", cex=1.5);
  points(Z[y==i,i1], Z[y==i,i2], col=i, type="p", pch=21, cex=0.7);
}

## IV) Logistic regression, (4.17) on page 119, multinomial response
data(vowel.train)
# need to install the package "VGAM" first
library(VGAM)
fitvoweltrain <- vglm(y~x.1+x.2+x.3+x.4+x.5+x.6+x.7+x.8+x.9+x.10,
                      family=multinomial(), data=vowel.train)
summary(fitvoweltrain)
fitvoweltrain@coefficients
fitvoweltrain@fitted.values  

## check training errors, Table 4.1, page 107
fittedlabel <- apply(fitvoweltrain@fitted.values, 1, which.max)
sum(fittedlabel!=vowel.train$y) # 118
sum(fittedlabel!=vowel.train$y)/dim(vowel.train)[1]  # 0.2234848
par(mfrow=c(1,1))
i1 <- 1
i2 <- 2
plot(Z[,i1], Z[,i2], xlab="Coordinate 1 for Training Data", ylab="Coordinate 2 for Training Data", 
     main="Linear Discriminant Analysis", type="n")
for(i in 1:11) points(Mnew[i,i1], Mnew[i,i2], col=i, type="p", pch="o", cex=2);
text(Z[fittedlabel<10,i1], Z[fittedlabel<10,i2], labels=fittedlabel[fittedlabel<10], col=fittedlabel[fittedlabel<10], cex=0.7) 
text(Z[fittedlabel==10,i1], Z[fittedlabel==10,i2], labels="0", col=10, cex=0.7) 
text(Z[fittedlabel==11,i1], Z[fittedlabel==11,i2], labels="a", col=11, cex=0.7) 

## predict class labels of testing data 
data(vowel.test)
K <- 11              # 11 vowels or 11 classes
p <- 10              # dimension of input space
nt <- dim(vowel.test)[1]                # number of observations
yt <- vowel.test$y   # output or response, integer values 1, 2, ..., 11
Xt <- vowel.test[,2:(p+1)]   # input values
Xt1 <- cbind(rep(1, nt), Xt) # matrix [1 X], (462*11)
betafitted <- fitvoweltrain@coefficients
dim(betafitted) <- c(K-1,K)  # fitted beta values based on training data
fittedtestvalues <- matrix(1, nt, K)    # fitted Pr(u belongs to k^th class | x_u)
fittedtestvalues[,1:10] <- exp(as.matrix(Xt1) %*% t(as.matrix(betafitted)))
fittedtestvalues <- fittedtestvalues/apply(fittedtestvalues, 1, sum)
# testing error rate, Table 4.1, page 107
fittedlabelt <- apply(fittedtestvalues, 1, which.max)
sum(fittedlabelt!=vowel.test$y)  #  237
sum(fittedlabelt!=vowel.test$y)/nt  # 0.512987
# display results
logisticresult <- cbind(vowel.test$y, fittedlabelt, round(fittedtestvalues[cbind(seq(1:nt),fittedlabelt)],3), round(fittedtestvalues[cbind(seq(1:nt),vowel.test$y)],3))
colnames(logisticresult) <- c("True", "Fit", "FitProb", "TrueProb")
logisticresult

## plot training data and testing data
#postscript("vowel_train.ps",paper="special",width=8,height=6.5,horizontal=F)
par(mfrow=c(1,1))
i1 <- 1
i2 <- 2
plot(Z[,i1], Z[,i2], xlab="Coordinate 1 for Training Data", ylab="Coordinate 2 for Training Data", 
     main="Linear Discriminant Analysis", type="n")
for(i in 1:9) {
  text(Mnew[i,i1], Mnew[i,i2], labels=i, cex=2);
  text(Z[y==i,i1], Z[y==i,i2], labels=i, cex=0.8);
}
text(Mnew[10,i1], Mnew[10,i2], labels="0", cex=2);
text(Z[y==10,i1], Z[y==10,i2], labels="0", cex=0.8);
text(Mnew[11,i1], Mnew[11,i2], labels="a", cex=2);
text(Z[y==11,i1], Z[y==11,i2], labels="a", cex=0.8);
#dev.off()

Zold <- as.matrix(X) %*% V
Zt <- as.matrix(Xt) %*% V
for(i in 1:10) Zt[,i] <- Zt[,i] - mean(Zold[,i]);
Zt <- -Zt
Zt[,2] <- -Zt[,2]
Zt[,10] <- -Zt[,10]
#postscript("vowel_test.ps",paper="special",width=8,height=6.5,horizontal=F)
par(mfrow=c(1,1))
i1 <- 1
i2 <- 2
plot(append(Z[,i1],Zt[,i1]), append(Z[,i2],Zt[,i2]), xlab="Coordinate 1 for Training Data", ylab="Coordinate 2 for Training Data", 
     main="Training Centroids Plus Testing Data", type="n")
for(i in 1:9) {
  text(Mnew[i,i1], Mnew[i,i2], labels=i, cex=2);
  text(Zt[yt==i,i1], Zt[yt==i,i2], labels=i, cex=0.8);
}
text(Mnew[10,i1], Mnew[10,i2], labels="0", cex=2);
text(Zt[yt==10,i1], Zt[yt==10,i2], labels="0", cex=0.8);
text(Mnew[11,i1], Mnew[11,i2], labels="a", cex=2);
text(Zt[yt==11,i1], Zt[yt==11,i2], labels="a", cex=0.8);
#dev.off()
