



from random import sample


# 助手操作，展示
def AssistantOrdersCards(deck):
    choice = chooseCards(deck)
    firstResults = findFirst(choice)
    nextResults = nextCards(deck, choice, firstResults)
    firstCard = restoreData([choice[firstResults[0]]])
    hideCard = restoreData([choice[firstResults[1]]])
    displayCards = (deck[nextResults[0]], deck[nextResults[1]], deck[nextResults[2]])
    print("隐藏的牌为:", hideCard)
    print("展示的四张牌分别为:", firstCard[0], displayCards[0], displayCards[1], displayCards[2])
    return (firstCard[0], displayCards[0], displayCards[1], displayCards[2])
    
    
# 工具函数，选取五张牌，并进行数据转换
def chooseCards(deck):
    # 随机抽五张牌
    choice = sample(deck, 5)
    choice = transformData(choice)
    # 使返回的数据固定，调试用的
    # choice = [[3, 'D'], [4, 'H'], [6, 'C'], [5, 'C'], [11, 'H']]
    return choice
    
    
# 工具函数 将原始数据转换为更容易计算的形式
def transformData(cards):
    # 分割数据
    choice = [[s.split("_")[0], s.split("_")[1]] for s in cards]
    # 数据转换，将第一个数据转换为数字
    for i in range(len(choice)):
        if choice[i][0] == "A":
            choice[i][0] = 1
        elif choice[i][0] == "J":
            choice[i][0] = 11
        elif choice[i][0] == "Q":
            choice[i][0] = 12
        elif choice[i][0] == "K":
            choice[i][0] = 13
        else:     #已经是数字的了
            choice[i][0] = int(choice[i][0])
    return choice
    
    
# 工具函数，找出符号相同的两张牌，以及展示和隐藏的牌
def findFirst(cards):
    symbol = {'C':0, 'D':0, 'H':0, 'S':0}
    hide_card = ''
    for card in cards:
        symbol[card[1]] += 1
    key = max(symbol.keys(),key=(lambda x:symbol[x]))
    first = second = -1
    i = 0
    for card in cards:
        if card[1] == key:
            if first == -1:
                first = i
            elif second == -1:
                second = i
                break
        i += 1
        
    # 找出展示的牌和隐藏的牌。
    v_first = cards[first][0]
    v_second = cards[second][0]
    gap = 0
    if v_first < v_second:
        gap = v_second - v_first
        if gap > 6:
            first, second = second, first
            gap = 13 - gap
    else:
        gap = v_first - v_second
        if gap > 6:
            gap = 13 - gap
        else:
            first, second = second, first
    return (first, second, gap)
    
    
# 根据前两张牌的情况展示另外三张牌
def nextCards(deck, cards, firstResults):
    nextThree = []
    i = 0
    for card in cards:
        if i != firstResults[0] and i != firstResults[1]:
            nextThree.append(card)
        i += 1
    # 将数据恢复成原始的形式
    nextThree = restoreData(nextThree)
    index = []
    # 得到最后三张牌在原数据中的位置，用于区分大中小
    for item in nextThree:
        index.append(deck.index(item))
    result = displayOrder(index, firstResults[2])
    return result
    
    
# 根据gap值计算最后三张牌的展示顺序
def displayOrder(index, gap):
    index = sorted(index)
    small, mid, large = index[0], index[1], index[2]
    # 计算最后三张牌的展示顺序
    result = []
    if gap == 1: # 小中大
        result = [small, mid, large]
    elif gap == 2: # 小大中
        result = [small, large, mid]
    elif gap == 3: # 中小大
        result = [mid, small, large]
    elif gap == 4: # 中大小
        result = [mid, large, small]
    elif gap == 5: # 大小中
        result = [large, small, mid]
    elif gap == 6: # 大中小
        result = [large, mid, small]
    else:
        print("出错了 b")
    
    return result
    
    
# 工具函数，将数据还原会原来的样子。
def restoreData(cards):
    n = len(cards)
    for i in range(n):
        if cards[i][0] == 1:
            cards[i][0] = 'A'
        elif cards[i][0] == 11:
            cards[i][0] = 'J'
        elif cards[i][0] == 12:
            cards[i][0] = 'Q'
        elif cards[i][0] == 13:
            cards[i][0] = 'K'
        else:
            cards[i][0] = str(cards[i][0])
    result = []
    for card in cards:
        s = '_'.join(card)
        result.append(s)
    return result
    
    
# 魔术师根据cards里展示的四张牌猜牌
def MagicianGuessesCard(deck, cards):
    new_cards = [card for card in cards]
    new_cards = transformData(new_cards)
    # 先确定牌的符号，跟第一张牌一样
    symble = new_cards[0][1]
    # 下面确定牌的数值
    gap = getGap(deck, cards)
    # 计算隐藏的牌的数值
    num = new_cards[0][0] + gap
    if num > 13:
        num = num - 13
    # 数值和符号都有了，可以合并得到结果了。
    result = [[num, symble]]
    result = restoreData(result)
    return result
    
    
# 工具函数，根据牌的顺序确定前两张牌的间隔
def getGap(deck, cards):
    index = []
    for i in range(1, 4):
        index.append(deck.index(cards[i]))
    new_index = sorted(index)
    small, mid, large =new_index[0], new_index[1], new_index[2]
    # 根据三张牌排列情况计算gap值
    if index == [small, mid, large]: # 小中大
        gap = 1
    elif index == [small, large, mid]: # 小大中
        gap = 2
    elif index == [mid, small, large]: # 中小大
        gap = 3
    elif index == [mid, large, small]: # 中大小
        gap = 4
    elif index == [large, small, mid]: # 大小中
        gap = 5
    elif index == [large, mid, small]: # 大中小
        gap = 6
    else:
        print("出错了 a")
    return gap
    


if __name__ == "__main__":
    deck = ['A_C','A_D','A_H','A_S',
                  '2_C','2_D','2_H','2_S',
                  '3_C','3_D','3_H','3_S',
                  '4_C','4_D','4_H','4_S',
                  '5_C','5_D','5_H','5_S',
                  '6_C','6_D','6_H','6_S',
                  '7_C','7_D','7_H','7_S',
                  '8_C','8_D','8_H','8_S',
                  '9_C','9_D','9_H','9_S',
                  '10_C','10_D','10_H','10_S', 
                  'J_C','J_D','J_H','J_S',
                  'Q_C','Q_D','Q_H','Q_S', 
                  'K_C','K_D','K_H','K_S']
    cards = AssistantOrdersCards(deck)
    result = MagicianGuessesCard(deck, cards)
    print("猜牌结果为:", result)
    