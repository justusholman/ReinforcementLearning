rm(list = ls())

library(MASS) 
library(reshape2) 
library(reshape) 
library(contextual)
library(ggnormalviolin)
library(factoextra)
library(fastDummies)
library(FactoMineR)
library("dplyr")
if (!require("pacman")) install.packages("pacman")
pacman::p_load(dplyr,
               tidyr,
               ggplot2,
               reshape2,
               latex2exp,
               devtools,
               BiocManager)

get_results <- function(history){
  df <- history$data %>%
    select(t, sim, choice, reward, agent) 
  return(df)
}


get_maxObsSims <- function(dfResults){
  dfResults_max_t <- dfResults %>%
    group_by(sim) %>% # group by per agent
    summarize(max_t = max(t)) # get max t
  return(dfResults_max_t)
}


show_results_multipleagents <- function(df_results,max_obs=900){
  ## Plot avg cumulative reward
  # Max of observations. Depends on the number of observations per simulation
  # Maximum number of observations
  
  # data.frame aggregated for two versions: 20 and 40 arms
  df_history_agg <- df_results %>%
    group_by(agent, sim)%>% # group by number of arms, the sim
    mutate(cumulative_reward = cumsum(reward))%>% # calculate cumulative sum
    group_by(agent, t) %>% # group by number of arms, the t
    summarise(avg_cumulative_reward = mean(cumulative_reward),# calc cumulative reward, se, CI
              se_cumulative_reward = sd(cumulative_reward, na.rm=TRUE)/sqrt(n_sim)) %>%
    mutate(cumulative_reward_lower_CI =avg_cumulative_reward - 1.96*se_cumulative_reward,
           cumulative_reward_upper_CI =avg_cumulative_reward + 1.96*se_cumulative_reward)%>%
    filter(t <=max_obs)
  
  
  # create ggplot object
  fig1 <- ggplot(data=df_history_agg, aes(x=t, y=avg_cumulative_reward, color =agent))+
    geom_line(size=1.5)+
    geom_ribbon(aes(ymin=cumulative_reward_lower_CI , 
                    ymax=cumulative_reward_upper_CI,
                    fill = agent,
    ),
    alpha=0.1)+
    labs(x = 'Time', y='Cumulative Reward', color ='c', fill='c')+
    theme_bw()+
    theme(text = element_text(size=16))
  
  ## arms plot
  fig2 <- ggplot(df_results, aes(x=choice)) + 
    geom_bar(color = as.numeric(sort(unique(df_results$choice))), fill = 
               as.numeric(sort(unique(df_results$choice)))) + 
    labs(title="", x="Item id", y = 'Number of selections') + 
    theme_minimal()
  return(list(fig1 = fig1, fig2 = fig2))
}


df <- read.csv("dataset.csv",header=TRUE)
df <- na.omit(df)

nobs <- dim(df)[1]-1
dfreturns <- 100*(log(df[-1,2:5]) - log(df[-dim(df)[1],2:5]))
dfreturnlags <- dfreturns[10:(nobs-1),]
colnames(dfreturnlags) <- c("AAPLlag","AMZNlag","GOOGLlag","MSFTlag")
dfreturncurrent <- dfreturns[11:nobs,]
dfreturnavg <- matrix(,nrow=nobs-10,ncol=4)
for (i in 1:(nobs-10)){
  dfreturnavg[i,] <- colSums(dfreturns[((i-1)*10+1):(i*10),])
}
colnames(dfreturnavg) <- c("AAPLavg","AMZNavg","GOOGLavg","MSFTavg")

df <- df[11:nobs,]
df[,2:5] <- dfreturncurrent
df <- cbind(df,dfreturnlags,dfreturnavg)

vLin <- seq(-1,1,0.5)
mPortComp <- tibble(x1=vLin,x2=vLin,x3=vLin,x4=vLin) %>% 
  expand(x1, x2, x3, x4) %>%
  filter(x1!=x2 & x2!=x3 & x3!=x4 )
mPortComp <- as.matrix(mPortComp[mPortComp[,1] + mPortComp[,2] + mPortComp[,3] + mPortComp[,4] == 1,])

narms <- dim(mPortComp)[1]
nobs <- dim(df)[1]

data <- matrix(,nrow=nobs*narms,ncol=6)
dim(data)
for (i in 1:nobs){
  data[((i-1)*narms+1):(i*narms),1] <- rep(df[i,1],narms)
  data[((i-1)*narms+1):(i*narms),2] <- c(1:narms)
  data[((i-1)*narms+1):(i*narms),3] <- mPortComp %*% as.numeric(df[i,2:5])
  data[((i-1)*narms+1):(i*narms),4] <- mPortComp %*% as.numeric(df[i,6:9])
  data[((i-1)*narms+1):(i*narms),5] <- mPortComp %*% as.numeric(df[i,10:13])
  data[((i-1)*narms+1):(i*narms),6] <- mPortComp %*% as.numeric(df[i,14:17])
}
dim(data)
colnames(data) <- c("time","arm","return","volume","lag","avg")
dfres <- data.frame(data)
write.csv(dfres,"C:\\Users\\justu\\OneDrive\\Documents\\aTIYear2\\Reinforcement_Learning\\Project\\dataneg.csv")

data <- read.csv("data.csv",header=TRUE)
dataneg <- read.csv("dataneg.csv",header=TRUE)
datagauss <- read.csv("datagauss.csv",header=TRUE)



size_sim=100000
n_sim=10
banditUCB <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = data, randomize = FALSE)
banditTS <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = data, randomize = FALSE)

UCB <- LinUCBDisjointPolicy$new(alpha=5)
TS <- ThompsonSamplingPolicy$new()

agentUCB <- Agent$new(UCB, banditUCB, name="UCB")
agentTS <- Agent$new(TS, banditTS, name="TS")

simulator <- Simulator$new(list(agentTS), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,3000)




size_sim=100000
n_sim=10
banditUCB <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = data, randomize = FALSE)
banditTS <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = data, randomize = FALSE)
banditUCBcontext <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag + avg, data = data, randomize = FALSE)
banditTScontext <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag + avg, data = data, randomize = FALSE)

UCB <- LinUCBDisjointPolicy$new(alpha=5)
TS <- ThompsonSamplingPolicy$new()
UCBcontext <- LinUCBDisjointPolicy$new(alpha=5)
TScontext <- ContextualLinTSPolicy$new(v=1)

agentUCB <- Agent$new(UCB, banditUCB, name="UCB")
agentTS <- Agent$new(TS, banditTS, name="TS")
agentUCBcontext <- Agent$new(UCBcontext, banditUCBcontext, name="contextUCB")
agentTScontext <- Agent$new(TScontext, banditTScontext, name="contextTS")

simulator <- Simulator$new(list(agentUCB,agentTS,agentUCBcontext,agentTScontext), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,500)




size_sim=100000
n_sim=10
banditUCB005 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = data, randomize = FALSE)
banditUCB01 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = data, randomize = FALSE)
banditUCB05 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = data, randomize = FALSE)
banditUCB1 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = data, randomize = FALSE)
banditUCB5 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = data, randomize = FALSE)

valpha=c(0.05, 0.1, 0.5, 1, 5)
UCB005 <- LinUCBDisjointPolicy$new(alpha=valpha[1])
UCB01 <- LinUCBDisjointPolicy$new(alpha=valpha[2])
UCB05 <- LinUCBDisjointPolicy$new(alpha=valpha[3])
UCB1 <- LinUCBDisjointPolicy$new(alpha=valpha[4])
UCB5 <- LinUCBDisjointPolicy$new(alpha=valpha[5])

agentUCB005 <- Agent$new(UCB005, banditUCB005, name="UCB 0.05")
agentUCB01 <- Agent$new(UCB01, banditUCB01, name="UCB 0.1")
agentUCB05 <- Agent$new(UCB05, banditUCB05, name="UCB 0.5")
agentUCB1 <- Agent$new(UCB1, banditUCB1, name="UCB 1")
agentUCB5 <- Agent$new(UCB5, banditUCB5, name="UCB 5")

simulator <- Simulator$new(list(agentUCB005,agentUCB01,agentUCB05,agentUCB1,agentUCB5), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,3000)



size_sim=100000
n_sim=10
banditTS001 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag + avg, data = data, randomize = FALSE)
banditTS005 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag + avg, data = data, randomize = FALSE)
banditTS01 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag + avg, data = data, randomize = FALSE)
banditTS05 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag + avg, data = data, randomize = FALSE)
banditTS1 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag + avg, data = data, randomize = FALSE)

vV =c(0.01, 0.05, 0.1, 0.5, 1)
TS001 <- ContextualLinTSPolicy$new(v=vV[1])
TS005 <- ContextualLinTSPolicy$new(v=vV[2])
TS01 <- ContextualLinTSPolicy$new(v=vV[3])
TS05 <- ContextualLinTSPolicy$new(v=vV[4])
TS1 <- ContextualLinTSPolicy$new(v=vV[5])

agentTS001 <- Agent$new(TS001, banditTS001, name="TS 0.01")
agentTS005 <- Agent$new(TS005, banditTS005, name="TS 0.05")
agentTS01 <- Agent$new(TS01, banditTS01, name="TS 0.1")
agentTS05 <- Agent$new(TS05, banditTS05, name= "TS 0.5")
agentTS1 <- Agent$new(TS1, banditTS1, name="TS 1")

simulator <- Simulator$new(list(agentTS001, agentTS005,agentTS01,agentTS05,agentTS1), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,1500)




vN <- c(20000,50000,100000)
vsim = c(2,5,10,20)
alpha = 5
banditUCB <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm , data = data, randomize = FALSE)
banditTS <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = data, randomize=FALSE)

UCB <- LinUCBDisjointPolicy$new(alpha=alpha)
TS <- ThompsonSamplingPolicy$new()

agentUCB <- Agent$new(UCB, banditUCB, name="UCB")
agentTS <- Agent$new(TS, banditTS, name="TS")

simulator <- Simulator$new(list(agentUCB,agentTS), # set our agents
                           horizon= vN[3], # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = vsim[1], # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,2500)




size_sim=50000
n_sim=10
banditUCB <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = dataneg, randomize = FALSE)
banditTS <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = dataneg, randomize = FALSE)
banditUCBcontext <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag + avg, data = dataneg, randomize = FALSE)
banditTScontext <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag + avg, data = dataneg, randomize = FALSE)

UCB <- LinUCBDisjointPolicy$new(alpha=5)
TS <- ThompsonSamplingPolicy$new()
UCBcontext <- LinUCBDisjointPolicy$new(alpha=5)
TScontext <- ContextualLinTSPolicy$new(v=1)

agentUCB <- Agent$new(UCB, banditUCB, name="UCB")
agentTS <- Agent$new(TS, banditTS, name="TS")
agentUCBcontext <- Agent$new(UCBcontext, banditUCBcontext, name="contextUCB")
agentTScontext <- Agent$new(TScontext, banditTScontext, name="contextTS")

simulator <- Simulator$new(list(agentUCB,agentTS), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,1300)





size_sim=100000
n_sim=10
banditUCB <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = data, randomize = FALSE)
banditUCB1 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | lag, data = data, randomize = FALSE)
banditUCB2 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | avg, data = data, randomize = FALSE)
banditUCB3 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume, data = data, randomize = FALSE)
banditUCB4 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag, data = data, randomize = FALSE)
banditUCB5 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | lag + avg, data = data, randomize = FALSE)
banditUCB6 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + avg, data = data, randomize = FALSE)
banditUCB7 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag + avg, data = data, randomize = FALSE)

alpha <- 5
UCB <- LinUCBDisjointPolicy$new(alpha=alpha)
UCB1 <- LinUCBDisjointPolicy$new(alpha=alpha)
UCB2 <- LinUCBDisjointPolicy$new(alpha=alpha)
UCB3 <- LinUCBDisjointPolicy$new(alpha=alpha)
UCB4 <- LinUCBDisjointPolicy$new(alpha=alpha)
UCB5 <- LinUCBDisjointPolicy$new(alpha=alpha)
UCB6 <- LinUCBDisjointPolicy$new(alpha=alpha)
UCB7 <- LinUCBDisjointPolicy$new(alpha=alpha)

agentUCB <- Agent$new(UCB, banditUCB, name="UCB")
agentUCB1 <- Agent$new(UCB1, banditUCB1, name="UCB Lag")
agentUCB2 <- Agent$new(UCB2, banditUCB2, name="UCB Average")
agentUCB3 <- Agent$new(UCB3, banditUCB3, name="UCB Volume")
agentUCB4 <- Agent$new(UCB4, banditUCB4, name="UCB vollag")
agentUCB5 <- Agent$new(UCB5, banditUCB5, name="UCB lagavg")
agentUCB6 <- Agent$new(UCB6, banditUCB6, name="UCB volavg")
agentUCB7 <- Agent$new(UCB7, banditUCB7, name="UCB all")

simulator <- Simulator$new(list(agentUCB,agentUCB1,agentUCB2,agentUCB3,agentUCB4,agentUCB5,agentUCB6,agentUCB7), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,1500)





size_sim=100000
n_sim=10
banditTS <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm, data = data, randomize = FALSE)
banditTS1 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | lag, data = data, randomize = FALSE)
banditTS2 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | avg, data = data, randomize = FALSE)
banditTS3 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume, data = data, randomize = FALSE)
banditTS4 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag , data = data, randomize = FALSE)
banditTS5 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | lag + avg, data = data, randomize = FALSE)
banditTS6 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + avg, data = data, randomize = FALSE)
banditTS7 <- OfflineReplayEvaluatorBandit$new(formula = return ~ arm | volume + lag + avg, data = data, randomize = FALSE)

alpha <- 5
TS <- ContextualLinTSPolicy$new(v=1)
TS1 <- ContextualLinTSPolicy$new(v=1)
TS2 <- ContextualLinTSPolicy$new(v=1)
TS3 <- ContextualLinTSPolicy$new(v=1)
TS4 <- ContextualLinTSPolicy$new(v=1)
TS5 <- ContextualLinTSPolicy$new(v=1)
TS6 <- ContextualLinTSPolicy$new(v=1)
TS7 <- ContextualLinTSPolicy$new(v=1)

agentTS <- Agent$new(TS, banditTS, name="TS")
agentTS1 <- Agent$new(TS1, banditTS1, name="TS Lag")
agentTS2 <- Agent$new(TS2, banditTS2, name="TS Average")
agentTS3 <- Agent$new(TS3, banditTS3, name="TS Volume")
agentTS4 <- Agent$new(TS4, banditTS4, name="TS vollag")
agentTS5 <- Agent$new(TS5, banditTS5, name="TS lagag")
agentTS6 <- Agent$new(TS6, banditTS6, name="TS volavg")
agentTS7 <- Agent$new(TS7, banditTS7, name="TS all")

simulator <- Simulator$new(list(agentTS,agentTS1,agentTS2,agentTS3,agentTS4,agentTS5,agentTS6,agentTS7), # set our agents
                           horizon= size_sim, # set the sizeof each simulation
                           do_parallel = TRUE,
                           worker_max = 12,# run in parallel for speed
                           simulations = n_sim, # simulate it n_sim times,
)
# run the simulator object
history_coins <- simulator$run()
res <- get_results(history_coins)
num <- get_maxObsSims(res)
print(num)
show_results_multipleagents(res,1750)
