eth.day <- ETH_USD_Daily
View(eth.day)
summary (eth.day)

#RUN SUMMARY FOR X VARIABLE
summary(eth.day$Open)
summary (eth.day$High)
summary (eth.day$Low)

#CORRELATION MATRIX
round(cor(eth.day),2) #to round off the number
eth.day <- eth.day[, -1]
eth.day<- eth.day[,-5:-6] #buang variable that is uncorrelated
cor(eth.day)
pairs(~Close+Open+High+Low, data=eth.day)

#LINEAR REGRESSION
model_1 <- lm(Close~Open+High+Low, data=eth.day)
plot(Close~Open+High+Low, data=eth.day)
abline (model_1)
summary(model_1)

# CREATE TRAINING AND TEST DATA 
set.seed(100)  # setting seed to reproduce results of random sampling
trainingRowIndex <- sample(1:nrow(eth.day), 0.8*nrow(eth.day))  # row indices for training data
trainingData <- eth.day[trainingRowIndex, ]  # model training data
testData  <- eth.day[-trainingRowIndex, ]   # test data

# FIT THE MODEL ON TRAINING DATA AND PREDICT CLOSE ON TEST DATA
lmMod <- lm(Close~Open+High+Low, data=trainingData)  # build the model
distPred <- predict(lmMod, testData)  # predict close price


#REVIEW DIAGNOSTICS MEASURES
summary (lmMod)  # model summary

# CALCULATE PREDICTION ACCURACY AND ERROR RATES
actuals_preds <- data.frame(cbind(actuals=testData$Close, predicteds=distPred))  # make actuals_predicteds dataframe.
correlation_accuracy <- cor(actuals_preds)
head(actuals_preds)

#PLOT OF PREDICTED VS ACTUAL VALUES
plot(x=distPred, y=testData$Close, xlab = 'Predicted Values', ylab = 'Actual Values')
abline(a=0,b=1)

# Min-Max Accuracy Calculation
min_max_accuracy <- mean(apply(actuals_preds, 1, min) / apply(actuals_preds, 1, max))

# MAPE Calculation
mape <- mean(abs((actuals_preds$predicteds - actuals_preds$actuals))/actuals_preds$actuals)

AIC(model_1)
BIC (model_1)

AIC(lmMod)
BIC(lmMod)

###############
#CODING USING DAILY DATA
library(DBI)
library(corrgram)
library(caret)
library(gridExtra)
library(ggpubr)
library(ggplot2)
library(forecast)
library(xts)
library(dygraphs)
library(caTools)
library(tseries)
library(zoom)
library(doParallel)

cl <- makeCluster(detectCores(), type= 'PSOCK')
registerDoParallel(cl)

setwd("C:/Users/Personal/Desktop")
getwd()

data <- read.csv("ETH-USD Monthly.csv", header = TRUE, sep = ',')
colnames(data)

head(data, 10)
tail(data, 10)
dim(data)
table(unlist(lapply(data, class)))

data <- data[,-7]
data <- data[,-6]
plot.ts(data[,c(4)])     
plot.ts(data["Close"])

data_new <- data                                          
data_new$Date <- as.Date(data_new$Date, format = "%Y-%m-%d")                 
data_new <- data_new[order(data_new$Date), ]           
data_new                                                  
plot(data_new$Date,                                     
     data_new$Close,
     type = "l",
     xaxt = "n")
axis(1,                                                   
     data_new$Date,
     format(data_new$Date, "%d/%m/%Y"))


eth_data <- as.numeric(data_new[,c("Close")])
eth.data <- data_new

fit_nnetar <- nnetar(eth.data$Close, repeats = 100 , size = 3)
print(fit_nnetar)
checkresiduals(fit_nnetar)

r <- cor(fitted(fit_nnetar)[14:length(eth_data)], 
         eth_data[14:length(eth_data)])
r2 <- cor(fitted(fit_nnetar)[14:length(eth_data)], 
          eth_data[14:length(eth_data)])^2
print(r2)
print(r)

x <- eth_data           
y <- fitted(fit_nnetar) 
ts.plot(x,y,
        gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

fitted(fit_nnetar)
#plotting Observed and Predicted with CI
accuracy(fitted(fit_nnetar),eth_data) 

forecast_ <- forecast:::forecast.nnetar(fit_nnetar, h = 50, 
                                        level = c(75,95), PI = TRUE)
autoplot(forecast_)

par(mfrow = c(1,1))
plot(forecast_)
summary (forecast_)

head(fitted(fit_nnetar))

#TESTING AND VALIDATION
data <- read.csv("ETH_DATA.csv", header = TRUE, sep = ',')
colnames(data)
data$Date <- as.Date(data$Date, format = "%d/%m/%Y")
str(data)
featurePlot(x=data[,2:4],y=data$Close,plot='pairs')
data <- data[,-c(7)]
data <- data[,-c(6)]

CloseTS <- ts(data$Close, start=c(2017, 1), frequency=365.25)
time <- time(CloseTS)
CloseTS
plot(data$Date,                                     
    CloseTS,
    type = "l",main = "Close Price", xlab="Date",  ylab="Close", bty="l", xaxt="n", yaxt="n")
axis(1,                                                   
    data$Date,
    format(data$Date, "%d/%m/%Y"))

validLength <- 292
trainLength <- length(CloseTS) - validLength
CloseTrain <- window(CloseTS, end=time[trainLength])
CloseValid <- window(CloseTS, start=time[trainLength+1])
CloseTrain
CloseValid

# Use nnetar to fit the neural network.
set.seed(227)
CloseNN <- nnetar(CloseTrain, repeats = 100, P=1,size = 3)
CloseNN
CloseNN.pred <- forecast(CloseNN, h = validLength)
accuracy(CloseNN.pred, CloseValid)
CloseNN.pred

# Set up the plot
plot(CloseTrain, ylim = c(50, 3000), main = "Ethereum Close Price", ylab = "Close Price", xlab = "Year", bty = "l", xaxt = "n", xlim =c(2017,2022), lty = 1)
axis(1, at = seq(2017,2022,1), labels = format(seq(2017,2022,1)))

lines(CloseNN.pred$fitted, lwd = 2, col = "red")
lines(CloseNN.pred$mean, lwd = 2, col = "red", lty = 2)
lines(CloseValid)
abline(v = 2020.25, col = "black", lty = 1, lwd = 1)
abline(h = 3000, col = "black", lty = 2, lwd = 1)
mtext("Training", line = -.5, at = c(2019,3200)) 
mtext("Validation", line = -.5, at = c(2021,3200))

# Plot the errors for the training period
plot(CloseNN.pred$residuals, 
    main = "Residual Plot for Training Period")
CloseNN.pred
checkresiduals(CloseNN.pred)
fitted(CloseNN)
plot(fitted(CloseNN))



#CODING USING MONTHLY DATA
cl <- makeCluster(detectCores(), type= 'PSOCK')
registerDoParallel(cl)
setwd("C:/Users/Personal/Desktop")
getwd()
data <- read.csv("ETH-USD Monthly.csv", header = TRUE, sep = ',')
colnames(data)

head(data, 10)
tail(data, 10)
dim(data)
table(unlist(lapply(data, class)))

plot.ts(data[,c(4)])     
plot.ts(data["Close"])

data <- data[,-c(7)]
data <- data[,-c(6)]

data_new <- data                                          
data_new$Date <- as.Date(data_new$Date, format = "%Y-%m-%d") #For daily data, format = "%d/%m/%Y                
data_new <- data_new[order(data_new$Date), ]           
data_new                                                  
plot(data_new$Date,                                     
    data_new$Close,
    type = "l",
    xaxt = "n")
axis(1,                                                   
    data_new$Date,
    format(data_new$Date, "%d/%m/%Y"))

eth_data <- as.numeric(data_new[,c("Close")])
eth.data <- data_new
fit_nnetar <- nnetar(eth.data$Close, repeats = 100 ,P = 1, size = 3)
print(fit_nnetar)

checkresiduals(fit_nnetar)

r <- cor(fitted(fit_nnetar)[14:length(eth_data)], 
        eth_data[14:length(eth_data)])
r2 <- cor(fitted(fit_nnetar)[14:length(eth_data)], 
         eth_data[14:length(eth_data)])^2
print(r2)

x <- eth_data           
y <- fitted(fit_nnetar) 
ts.plot(x,y,
       gpars = list(col = c("black", "red")))
legend("topleft", legend = c("Actual", "Predicted"), col = c("black", "red"), lty = 1)

accuracy(eth_data,fitted(fit_nnetar))

forecast_ <- forecast:::forecast.nnetar(fit_nnetar, h = 25, 
                                       level = c(75,95), PI = TRUE)
autoplot(forecast_)
summary (forecast_)
forecast_

#TRAINING VALIDATION
data <- read.csv("ETH-USD Monthly.csv", header = TRUE, sep = ',')
colnames(data)
data$Date <- as.Date(data$Date, format = "%Y-%m-%d")
data <- data[,-c(7)]
data <- data[,-c(6)]

CloseTS <- ts(data$Close, start=c(2017, 4), frequency=12)
yrange = range(CloseTS)
time <- time(CloseTS)
CloseTS

plot(c(2017, 2022), yrange, type="n", main = "Close Price", xlab="Date",  ylab="Close", bty="l", xaxt="n", yaxt="n")
lines(CloseTS, bty ="l")
axis(1, at=seq(2016,2021,1), labels=format(seq(2016,2021,1)))
axis(2, at=seq(50,3000,500), las=2)

validLength <- 10
trainLength <- length(CloseTS) - validLength
CloseTrain <- window(CloseTS, end=time[trainLength])
CloseValid <- window(CloseTS, start=time[trainLength+1])
CloseTrain
CloseValid
set.seed(227)
CloseNN <- nnetar(CloseTrain, P=1,size = 3)
CloseNN.pred <- forecast(CloseNN, h = validLength)
accuracy(CloseNN.pred, CloseValid)

plot(CloseTrain, ylim = c(50, 3000), main = "Ethereum Close Price", ylab = "Close Price", xlab = "Year", bty = "l", xaxt = "n", xlim =c(2017,2022), lty = 1)
axis(1, at = seq(2017,2022,1), labels = format(seq(2017,2022,1)))
lines(CloseNN.pred$fitted, lwd = 2, col = "red")
lines(CloseNN.pred$mean, lwd = 2, col = "red", lty = 2)
lines(CloseValid)
abline(v = 2020.5, col = "black", lty = 1, lwd = 1)
abline(h = 3000, col = "black", lty = 2, lwd = 1)
mtext("Training", line = -.5, at = c(2019,3200)) 
mtext("Validation", line = -.5, at = c(2021.25,3200))



plot(CloseNN.pred$residuals, main = "Residual Plot for Training Period")
checkresiduals(CloseNN.pred)
CloseNN.pred

#########################
#R Coding for GARCH (1,1) Ethereum
library(ggplot2)
library(caTools)
library(dygraphs)
library(xts)
library(forecast)
library(fGarch)
library(tseries)
getwd()

data=read.csv("ETH_DATA.csv",header= TRUE)
df=data
df=df[,-c(2,7)]
df=xts(df[,-1],order.by=as.Date(df[,1],"%d/%m/%Y"))

m <- head(df, n=1463)
dygraph(m) %>%
  dyCandlestick()
da=data.frame(data$Date,data$Close)
class(da$data.Close)
da$data.Date = as.Date(da$data.Date, '%d/%m/%Y')
dev.off()
ggplot(data=da, aes(x=data.Date,y=as.numeric(data.Close))) + geom_line()
da$logClose= log(as.numeric(da$data.Close))
ggplot(data=da, aes(data.Date,as.numeric(logClose))) + geom_line()
da$sqrt= sqrt(da$logClose)
ggplot(data=da, aes(data.Date,as.numeric(sqrt))) + geom_line()
diffClose=diff(da$logClose)
newFrame=da[-c(1),]
acf(diffClose)

ggplot(data=newFrame, aes(data.Date,diffClose)) + geom_line()
adf.test(diffClose)
fit1 = auto.arima(da$sqrt, trace = TRUE, test = "kpss", ic = "bic")
Box.test(fit1$residuals, lag = 12, type = "Ljung-Box")
acf(fit1$residuals^2)
tsdisplay(fit1$residuals)
tsdiag(fit1)

# garch effect is there
model=garchFit(~garch(1,1), data=diff(da$sqrt))
summary (model)

###################
#R Coding for GARCH (1,1) Bitcoin
library(ggplot2)
library(caTools)
library(dygraphs)
library(xts)
library(forecast)
library(fGarch)
library(tseries)
getwd()

data=read.csv("BTC_DATA.csv",header= TRUE)

df=data
df=df[,-c(2,7)]
df=xts(df[,-1],order.by=as.Date(df[,1],"%d/%m/%Y"))
m <- head(df, n=1463)
dygraph(m) %>%
  dyCandlestick()
da=data.frame(data$Date,data$Close)

class(da$data.Close)
da$data.Date = as.Date(da$data.Date, '%d/%m/%Y')
dev.off()

ggplot(data=da, aes(x=data.Date,y=as.numeric(data.Close))) + geom_line()
da$logClose= log(as.numeric(da$data.Close))

ggplot(data=da, aes(data.Date,as.numeric(logClose))) + geom_line()
da$sqrt= sqrt(da$logClose)

ggplot(data=da, aes(data.Date,as.numeric(sqrt))) + geom_line()
diffClose=diff(da$logClose)
newFrame=da[-c(1),]
acf(diffClose)

ggplot(data=newFrame, aes(data.Date,diffClose)) + geom_line()
adf.test(diffClose)
fit1 = auto.arima(da$sqrt, trace = TRUE, test = "kpss", ic = "bic")

Box.test(fit1$residuals, lag = 12, type = "Ljung-Box")
acf(fit1$residuals^2)
tsdisplay(fit1$residuals)
tsdiag(fit1)

# garch effect is there
model=garchFit(~garch(1,1), data=diff(da$sqrt))
summary (model)
