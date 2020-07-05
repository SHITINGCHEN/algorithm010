# 给定两个单词（beginWord 和 endWord）和一个字典 wordList，找出所有从 beginWord 到 endWord 的最短转换序列。转换
# 需遵循如下规则： 
# 
#  
#  每次转换只能改变一个字母。 
#  转换后得到的单词必须是字典中的单词。 
#  
# 
#  说明: 
# 
#  
#  如果不存在这样的转换序列，返回一个空列表。 
#  所有单词具有相同的长度。 
#  所有单词只由小写字母组成。 
#  字典中不存在重复的单词。 
#  你可以假设 beginWord 和 endWord 是非空的，且二者不相同。 
#  
# 
#  示例 1: 
# 
#  输入:
# beginWord = "hit",
# endWord = "cog",
# wordList = ["hot","dot","dog","lot","log","cog"]
# 
# 输出:
# [
#   ["hit","hot","dot","dog","cog"],
#   ["hit","hot","lot","log","cog"]
# ]
#  
# 
#  示例 2: 
# 
#  输入:
# beginWord = "hit"
# endWord = "cog"
# wordList = ["hot","dot","dog","lot","log"]
# 
# 输出: []
# 
# 解释: endWord "cog" 不在字典中，所以不存在符合要求的转换序列。 
#  Related Topics 广度优先搜索 数组 字符串 回溯算法


# leetcode submit region begin(Prohibit modification and deletion)
import collections

"""思路
思路分析：
单词列表、起点、终点构成无向图，容易想到使用广度优先遍历找到最短路径。图中广度优先遍历要使用到：队列；
标记是否访问过的「布尔数组」或者「哈希表」visited，这里 key 是字符串，故使用哈希表；

重点：由于要记录所有的路径，广度优先遍历「当前层」到「下一层」的所有路径都得记录下来。
因此找到下一层的结点 wordA 以后，不能马上添加到 visited 哈希表里，还需要检查当前队列中未出队的单词是否还能与 wordA 建立联系；
广度优先遍历位于同一层的单词，即使有联系，也是不可以被记录下来的，这是因为同一层的连接肯定不是起点到终点的最短路径的边；
使用 BFS 的同时记录遍历的路径，形式：哈希表。哈希表的 key 记录了「顶点字符串」，哈希表的值 value 记录了 key 对应的字符串在广度优先遍历的过程中得到的所有后继结点列表 successors；
最后根据 successors，使用回溯算法（全程使用一份路径变量搜索所有可能结果的深度优先遍历算法）得到所有的最短路径。

作者：liweiwei1419
链接：https://leetcode-cn.com/problems/word-ladder-ii/solution/yan-du-you-xian-bian-li-shuang-xiang-yan-du-you--2/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
"""
class Solution:
    """
    如果开始就构建图，结果可能会超时，这里采用的方法是：先把 wordList 存入哈希表，然后是一边遍历，一边找邻居，进而构建图，这是解决这个问题的特点；
    每一层使用一个新的 nextLevelVisited 哈希表，记录当前层的下一层可以访问到哪些结点。
    直到上一层队列里的值都出队以后， nextLevelVisited 哈希表才添加到总的 visited 哈希表，这样记录当前结点和广度优先遍历到的子结点列表才不会遗漏。
    算法思想
    第 1 步：使用广度优先遍历找到终点单词，并且记录下沿途经过的所有结点，以邻接表形式存储；
    第 2 步：通过邻接表，使用回溯算法得到所有从起点单词到终点单词的路径。
    """
    from collections import defaultdict, deque
    def findLadders(self, beginWord: str, endWord: str, wordList: List[str]) -> List[List[str]]:
        if not endWord in wordList or not wordList:
            return []
        word_set, res = set(wordList), []
        successor = defaultdict(set)
        # 第一步：使用BFS构建邻接表
        found = self.bfs(beginWord, endWord, word_set, successor)
        if not found:
            return res
        # 第二步，基于successor，构建回溯算法得到所有最短路径
        path = [beginWord]
        self.dfs(beginWord, endWord, successor, path, res)
        return res

    def bfs(self, beginWord, endWord, word_set, successor):
        que = deque([beginWord])
        visited = set()
        visited.add(beginWord)
        found = False
        L = len(beginWord)
        next_level_visited = set()
        while que:
            qsize = len(que)
            for _ in range(qsize):
                cur_word = que.popleft()
                for j in range(L):
                    for k in range(ord('a'), ord('z')+1):
                        next_word = cur_word[:j] + chr(k) + cur_word[j+1:]
                        if next_word in word_set:
                            if next_word not in visited:
                                if next_word == endWord:
                                    found = True
                                if next_word not in next_level_visited:
                                    next_level_visited.add(next_word)
                                    que.append(next_word)
                                successor[cur_word].add(next_word)
            if found:
                break
            visited |= next_level_visited
            next_level_visited = set()
        return found

    def dfs(self, beginWord, endWord, successor, path, res):
        if beginWord == endWord:
            res.append(path[:])
            return
        if beginWord not in successor:
            return
        for next_word in successor[beginWord]:
            path.append(next_word)
            self.dfs(next_word, endWord, successor, path, res)
            path.pop()

# leetcode submit region end(Prohibit modification and deletion)
