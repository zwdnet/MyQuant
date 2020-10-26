# coding:utf-8
# 机器学习A-Z
# 第五部分 强化学习


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import math


if __name__ == "__main__":
    dataset = pd.read_csv("./data/UCB/Ads_CTR_Optimisation.csv")
    print(dataset)
    N = 10000
    d = 10
    ads_selected = []
    total_reward = 0
    for n in range(0, N):
        ad = random.randrange(d)
        ads_selected.append(ad)
        reward = dataset.values[n, ad]
        total_reward = total_reward + reward
        
    print(total_reward)
    plt.hist(ads_selected)
    plt.title("ads selected")
    plt.savefig("adshist.png")
    
    # 实现UCB
    N = 10000
    d = 10
    numbers_of_selections = [0]*d
    sums_of_rewards = [0]*d
    for n in range(0, N):
        ad = 0
        max_upper_bound = 0
        for i in range(0, d):
            if (numbers_of_selections[i] > 0):
                average_reward = sums_of_rewards[i] / numbers_of_selections[i]
                delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
                upper_bound = average_reward + delta_i
            else:
                upper_bound = 1e400
            if upper_bound > max_upper_bound:
                max_upper_bound = upper_bound
                ad = i
        ads_selected.append(ad)
        numbers_of_selections[ad] = numbers_of_selections[ad] + 1
        reward = dataset.values[n, ad]
        sums_of_rewards[ad] = sums_of_rewards[ad] + reward
        total_reward = total_reward + reward
        
    plt.hist(ads_selected)
    plt.title('Histogram of ads selections')
    plt.xlabel('Ads')
    plt.ylabel('Number of times each ad was selected')
    plt.savefig("ucb.png")
    