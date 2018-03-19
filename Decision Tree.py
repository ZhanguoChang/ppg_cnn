import copy
import math
from graphviz import  Digraph

#决策树类
class DecisionTree():
    path = r'\data.txt'    #数据路径
    contents  = []              #数据
    contTypes = ['buying','maint','doors','persons','lug-boot','safety'] #数据类型
    decisionTree = []           #生成的决策树
    dot = Digraph(comment = "Decision Tree")            #用于绘制树的模块

    #读取数据
    def inputData(self):
        with open(self.path,'r') as file:
            if file == None:
                exit('数据文件路径不存在，请先确认数据文件')
            contents = file.readlines()
            return contents

    #处理数据
    def handleData(self):
        conts = self.inputData()
        dic  = {}                    #创建字典来存储
        for cont in conts:
            content = cont.split(',')
            dic['buying'] = content[0]      #价格，共有4种值：vhigh,high,med,low
            dic['maint'] = content[1]       #养护费，共有4种值：vhigh,high,med,low
            dic['doors'] = content[2]       #车门数，共有4种值：2,3,4,5-more
            dic['persons'] = content[3]     #核定载人数，共有3种值：2,4,more
            dic['lug-boot'] = content[4]    #尾箱大小，共有3种值：small,med,big
            dic['safety'] = content[5]      #安全性，共有3种值：low,med,high
            dic['result'] = content[6][:-1]    #评判结果，共有4种值：unacc,acc,good,vgood
            #print(dic['result'])
            self.contents.append(copy.deepcopy(dic))
        #for i in self.contents:
            #print(i)

    #打印数据用于调试
    def printf(self):
        print(self.calcCond(self.contents,'buying'))
        print(self.calcCond(self.contents, 'maint'))
        print(self.calcCond(self.contents, 'doors'))
        print(self.calcCond(self.contents, 'persons'))
        print(self.calcCond(self.contents, 'lug-boot'))
        print(self.calcCond(self.contents, 'safety'))



    #id3算法
    def id3(self):

        flag = 1
        maxType = None
        judge = {}
        contTypesQueue = []
        contentsQueue = []
        contentsQueue.append([copy.deepcopy(self.contents)])      #将所有待扩展的结点加入一个队列
        contTypesQueue.append(copy.deepcopy(self.contTypes))      #将所有将用的信息加入一个队列


        while contTypesQueue or flag == 1  :
            flag = 0
            max = 0
            contents = contentsQueue[0]
            contentsQueue = contentsQueue[1:]
            if contents :       #当不为空时
                k = 0
                for content in contents:                    #按照宽度优先遍历所有节点
                    max = 0                                 #每次置零
                    contTypes = contTypesQueue[0]           # 第一个元素出队
                    contTypesQueue = contTypesQueue[1:]
                    for contType in contTypes:
                        judge[contType] = self.calcCond(content, contType)
                        if judge[contType] > max:  # 用于存储使熵值下降最快的信息名
                            max = judge[contType]
                            maxType = contType

                    j = len(self.contTypes) - len(contTypes)
                    if max > 0:                            #表示还需要继续往下扩展
                        self.decisionTree.append([maxType,j,k,0])
                        flag = 1
                        newContents = self.subContent(content,maxType)
                        contentsQueue.append(copy.deepcopy(newContents))
                        newtypes = copy.deepcopy(contTypes)
                        newtypes.remove(maxType)
                        for i in range(len(newContents)):
                            contTypesQueue.append(copy.deepcopy(newtypes))

                    else:
                        self.decisionTree.append([content[0]['result'], j,k, 1])
                        #k+=1
                    k+=1



    #计算某条件熵，参数分别为数据集合、计算的类型，返回平均互信息量

    def calcCond(self,surpConts,condType):
        H = 0
        12
        sum = len(surpConts)
        sum1, sum2, sum3, sum4 = 0, 0, 0, 0   #代表四个可选值，当只有三个可选值时，最后一个一直为0，对结果不会造成影响
        sumUnacc, sumAcc, sumGood, sumVgood = 0, 0, 0, 0
        sum1Unacc, sum1Acc, sum1Good, sum1Vgood = 0, 0, 0, 0
        sum2Unacc, sum2Acc, sum2Good, sum2Vgood = 0, 0, 0, 0
        sum3Unacc, sum3Acc, sum3Good, sum3Vgood = 0, 0, 0, 0
        sum4Unacc, sum4Acc, sum4Good, sum4Vgood = 0, 0, 0, 0

        # 计算不同条件的熵值时，定义不同的比较条件，最多四个比较条件
        if condType == 'buying':
            cond_1,cond_2,cond_3,cond_4 = 'vhigh','high','med','low'

        elif condType == 'maint':
            cond_1, cond_2, cond_3, cond_4 = 'vhigh', 'high', 'med', 'low'

        elif condType == 'doors':
            cond_1, cond_2, cond_3, cond_4 = '2', '3', '4', '5more'

        elif condType == 'persons':
            cond_1, cond_2, cond_3, cond_4 = '2', '4', 'more', None

        elif condType == 'lug-boot':
            cond_1, cond_2, cond_3, cond_4 = 'small', 'med', 'big', None

        elif condType == 'safety':
            cond_1, cond_2, cond_3, cond_4 = 'low', 'med', 'high', None

        for cont in surpConts:                #遍历元素集，求出各条件对应的熵值
            if cont[condType] == cond_1:            #当为第一个条件的时候，例如对于传入的condType为buying，cond_1就为vhigh
                sum1 += 1
                if cont['result'] == 'unacc':
                    sum1Unacc += 1
                elif cont['result'] == 'acc':
                    sum1Acc += 1
                elif cont['result'] == 'good':
                    sum1Good += 1
                elif cont['result'] == 'vgood':
                    sum1Vgood += 1

            elif cont[condType] == cond_2:          #当为第二个条件的时候，例如对于传入的condType为buying，cond_1就为high
                sum2 += 1
                if cont['result'] == 'unacc':
                    sum2Unacc += 1
                elif cont['result'] == 'acc':
                    sum2Acc += 1
                elif cont['result'] == 'good':
                    sum2Good += 1
                elif cont['result'] == 'vgood':
                    sum2Vgood += 1

            elif cont[condType] == cond_3:          #当为第三个条件的时候，例如对于传入的condType为buying，cond_1就为med
                sum3 += 1
                if cont['result'] == 'unacc':
                    sum3Unacc += 1
                elif cont['result'] == 'acc':
                    sum3Acc += 1
                elif cont['result'] == 'good':
                    sum3Good += 1
                elif cont['result'] == 'vgood':
                    sum3Vgood += 1

            elif cont[condType] == cond_4:          #当为第四个条件的时候，，例如对于传入的condType为buying，cond_1就为low; 对于核定在人数，车箱大小，安全性是没有这个条件的
                sum4 += 1
                if cont['result'] == 'unacc':
                    sum4Unacc += 1
                elif cont['result'] == 'acc':
                    sum4Acc += 1
                elif cont['result'] == 'good':
                    sum4Good += 1
                elif cont['result'] == 'vgood':
                    sum4Vgood += 1

        if sum1 != 0:                               #需要作判0操作
            if sum1Unacc != 0:
                part1 = (sum1Unacc / sum1) * (math.log2(sum1Unacc / sum1))
            else:
                part1 = 0

            if sum1Acc != 0:
                part2 = (sum1Acc / sum1) * (math.log2(sum1Acc / sum1))
            else:
                part2 = 0

            if sum1Good != 0:
                part3 = (sum1Good / sum1) * (math.log2(sum1Good / sum1))
            else:
                part3 = 0

            if sum1Vgood != 0:
                part4 = (sum1Vgood / sum1) * (math.log2(sum1Vgood / sum1))
            else:
                part4 = 0

        H1 = -1 * sum1 / sum * (part1 + part2 + part3 + part4)



        if sum2 != 0:                               #需要作判0操作
            if sum2Unacc != 0:
                part1 = (sum2Unacc / sum2) * (math.log2(sum2Unacc / sum2))
            else:
                part1 = 0

            if sum2Acc != 0:
                part2 = (sum2Acc / sum2) * (math.log2(sum2Acc / sum2))
            else:
                part2 = 0

            if sum2Good != 0:
                part3 = (sum2Good / sum2) * (math.log2(sum2Good / sum2))
            else:
                part3 = 0

            if sum2Vgood != 0:
                part4 = (sum2Vgood / sum2) * (math.log2(sum2Vgood / sum2))
            else:
                part4 = 0

        H2 = -1 * sum2 / sum * (part1 + part2 + part3 + part4)


        if sum3 != 0 :                               #需要作判0操作
            if sum3Unacc != 0:
                part1 = (sum3Unacc / sum3) * (math.log2(sum3Unacc / sum3))
            else:
                part1 = 0

            if sum3Acc != 0:
                part2 = (sum3Acc / sum3) * (math.log2(sum3Acc / sum3))
            else:
                part2 = 0

            if sum3Good != 0:
                part3 = (sum3Good / sum3) * (math.log2(sum3Good / sum3))
            else:
                part3 = 0

            if sum3Vgood != 0:
                part4 = (sum3Vgood / sum3) * (math.log2(sum3Vgood / sum3))
            else:
                part4 = 0

        H3 = -1 * sum3 / sum * (part1 + part2 + part3 + part4)


        if sum4 != 0:                               #需要作判0操作
            if sum4Unacc != 0:
                part1 = (sum4Unacc / sum4) * (math.log2(sum4Unacc / sum4))
            else:
                part1 = 0

            if sum4Acc != 0:
                part2 = (sum4Acc / sum4) * (math.log2(sum4Acc / sum4))
            else:
                part2 = 0

            if sum4Good != 0:
                part3 = (sum4Good / sum4) * (math.log2(sum4Good / sum4))
            else:
                part3 = 0

            if sum4Vgood != 0:
                part4 = (sum4Vgood / sum4) * (math.log2(sum4Vgood / sum4))
            else:
                part4 = 0

        H4 = -1 * sum4 / sum * (part1 + part2 + part3 + part4)

        sumUnacc = sum1Unacc + sum2Unacc + sum3Unacc + sum4Unacc
        sumAcc = sum1Acc + sum2Acc + sum3Acc+ sum4Acc
        sumGood = sum1Good + sum2Good + sum3Good + sum4Good
        sumVgood = sum1Vgood + sum2Vgood + sum3Vgood + sum4Vgood

        if sum != 0:
            if sumUnacc != 0:
                part1 = (sumUnacc / sum) * (math.log2(sumUnacc / sum))
            else:
                part1 = 0

            if sumAcc != 0:
                part2 = (sumAcc / sum) * (math.log2(sumAcc / sum))
            else:
                part2 = 0

            if sumGood != 0:
                part3 = (sumGood / sum) * (math.log2(sumGood / sum))
            else:
                part3 = 0

            if sumVgood != 0:
                part4 = (sumVgood / sum) * (math.log2(sumVgood / sum))
            else:
                part4 = 0

        H = -1 * (part1 + part2 + part3 + part4)
        H_ = H1 + H2 + H3 + H4
        return H - H_

    #去除某部分已经使用的信息，返回其子集
    def subContent(self,newContents,condType):
        newContents1 = []
        newContents2 = []
        newContents3 = []
        newContents4 = []

        if condType == 'buying':
            cond_1,cond_2,cond_3,cond_4 = 'vhigh','high','med','low'

        elif condType == 'maint':
            cond_1, cond_2, cond_3, cond_4 = 'vhigh', 'high', 'med', 'low'

        elif condType == 'doors':
            cond_1, cond_2, cond_3, cond_4 = '2', '3', '4', '5more'

        elif condType == 'persons':
            cond_1, cond_2, cond_3, cond_4 = '2', '4', 'more', None

        elif condType == 'lug-boot':
            cond_1, cond_2, cond_3, cond_4 = 'small', 'med', 'big', None

        elif condType == 'safety':
            cond_1, cond_2, cond_3, cond_4 = 'low', 'med', 'high', None

        for content in newContents:
            if content[condType] == cond_1:
                newContents1.append(content)
            elif content[condType] == cond_2:
                newContents2.append(content)
            elif content[condType] == cond_3:
                newContents3.append(content)
            elif content[condType] == cond_4:
                newContents4.append(content)
        if cond_4 == None or cond_4 == []:
            return copy.deepcopy([newContents1,newContents2,newContents3])

        return copy.deepcopy([newContents1,newContents2,newContents3,newContents4])


    #划出决策树
    def drawTree(self):
        #print('":"后面的值为1时为叶子结点，为0时表示可以继续扩展')
        p = 0
        i = 0
        allNode = []                    #存储所有结点
        allNode.append([])              #每个列表存储一层数据
        for node in self.decisionTree:
            if node[1] > p:
                print()
                p = node[1]
                allNode.append([])
                i+=1
            if node[2] == 0:
                allNode[i].append(' ')
            allNode[i].append(node[0])


            #print(node[0]+':'+str(node[3]), end=' ')
        for i in range(len(allNode)):
            allNode[i] = allNode[i][1:]            #先把开始的空格去除
            #print(allNode[i])

        self.dot.node('allNode[0][0]', allNode[0][0])   #先把根结点加入

        #整理所有信息，用于构造决策树时标志边的含义
        buying = ['vhigh', 'high', 'med', 'low']
        maint = ['vhigh', 'high', 'med', 'low']
        doors = ['2', '3', '4', '5more']
        persons = ['2', '4', 'more']
        lugboot = ['small', 'med', 'big']
        safety = ['low', 'med', 'high']
        allSign = {}
        allSign['buying'] = buying
        allSign['maint'] = maint
        allSign['doors'] = doors
        allSign['persons'] = persons
        allSign['lug-boot'] = lugboot
        allSign['safety'] = safety

        i,j,k,e = 0,0,0,0

        resultList = ['unacc','acc','good','vgood',' ']
        for i in range(1,len(allNode)):               #遍历其余结点画出决策树
            k = 0
            while k < len(allNode[i-1])-1 and allNode[i-1][k] in resultList:          #找出上一层第一个非叶子结点
                k += 1
            sign = allSign[allNode[i-1][k]]
            e = 0
            for j in range(len(allNode[i])):
                if allNode[i][j] == ' ':
                    k += 1
                    e = 0
                    while k < len(allNode[i-1])-1 and allNode[i-1][k] in resultList:
                        k+=1
                    sign = allSign[allNode[i - 1][k]]
                    continue
                self.dot.node('allNode[' + str(i) + '][' + str(j) + ']',allNode[i][j])          #将结点添加到图形
                #self.dot.edges(['A' + str(i - 1) + str(k)  + 'A' + str(i) +  str(j)])
                #self.dot.edges(['allNode[' + str(i-1) + '][' + str(k) + ']' + 'allNode[' + str(i) + '][' + str(j) + ']'])
                self.dot.edge('allNode[' + str(i-1) + '][' + str(k) + ']',                      #画出其与父结点的连线
                              'allNode[' + str(i) + '][' + str(j) + ']',label=sign[e]
                              )
                e+=1

        #print(self.dot.source)
        self.dot.render('./Decison.gv', view=True)



if '__main__' == __name__:
    obj = DecisionTree()
    obj.handleData()
    obj.id3()
    obj.drawTree()

