# 给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度。转换需遵循如下规则：
#  
# 
#  
#  每次转换只能改变一个字母。 
#  转换过程中的中间单词必须是字典中的单词。 
#  
# 
#  说明: 
# 
#  
#  如果不存在这样的转换序列，返回 0。 
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
# 输出: 5
# 
# 解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",
#      返回它的长度 5。
#  
# 
#  示例 2: 
# 
#  输入:
# beginWord = "hit"
# endWord = "cog"
# wordList = ["hot","dot","dog","lot","log"]
# 
# 输出: 0
# 
# 解释: endWord "cog" 不在字典中，所以无法进行转换。 
#  Related Topics 广度优先搜索


# leetcode submit region begin(Prohibit modification and deletion)
from collections import deque
class Solution:
    """官方题解
    将问题抽象为一个无向无权图，每个单词为一个结点，
    相邻的结点是只有一个字母改变的单词，因此问题转化为找最短路径——BFS
    算法中最重要的步骤是找出相邻的节点，也就是只差一个字母的两个单词。
    为了快速的找到这些相邻节点，对给定的 wordList 做一个预处理，将单词中的某个字母用 * 代替。
    这个预处理帮我们构造了一个单词变换的通用状态。例如：Dog ----> D*g <---- Dig，Dog 和 Dig 都指向了一个通用状态 D*g。
    这步预处理找出了单词表中所有单词改变某个字母后的通用状态，并帮助我们更方便也更快的找到相邻节点。
    算法：
    1、对给定的 wordList 做预处理，找出所有的通用状态。将通用状态记录在字典中，键是通用状态，值是所有具有通用状态的单词。
    2、将包含 beginWord 和 1 的元组放入队列中，1 代表节点的层次。我们需要返回 endWord 的层次也就是从 beginWord 出发的最短距离
    3、为了防止出现环，使用访问数组记录。
    4、当队列中有元素的时候，取出第一个元素，记为 current_word。
    5、找到 current_word 的所有通用状态，并检查这些通用状态是否存在其它单词的映射，这一步通过检查 all_combo_dict 来实现。
    6、从 all_combo_dict 获得的所有单词，都和 current_word 共有一个通用状态，所以都和 current_word 相连，因此将他们加入到队列中。
    7、对于新获得的所有单词，向队列中加入元素 (word, level + 1) 其中 level 是 current_word 的层次。
    8、最终当你到达期望的单词，对应的层次就是最短变换序列的长度。
    """
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        if not endWord in wordList or not endWord or not beginWord or not wordList:
            return 0
        all_comb_dict = collections.defaultdict(list)
        L = len(beginWord)
        for word in wordList:
            for i in range(L):
                all_comb_dict[word[:i] + '*' + word[i+1:]].append(word)

        que, visited = deque([(beginWord, 1)]), {beginWord: True}
        while que:
            cur_word, step = que.popleft()
            for i in range(L):
                tmp_word = cur_word[:i] + '*' + cur_word[i+1:]
                for word in all_comb_dict[tmp_word]:
                    if word == endWord:
                        return step + 1
                    if word not in visited:
                        visited[word] = True
                        que.append((word, step + 1))
                all_comb_dict[tmp_word] = []
        return 0
    # 自己的思路
    # def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
    #
    #     wordList = set(wordList)
    #     if endWord not in wordList:
    #         return 0
    #     dic = beginWord
    #     for word in wordList:
    #         dic += word
    #     dic = set(dic)
    #     que = deque([(beginWord, 1)])
    #     while que:
    #         word, step = que.popleft()
    #         if word == endWord:
    #             return step
    #         for i, s in enumerate(word):
    #             for c in dic:
    #                 if c == s:
    #                     continue
    #                 new_word = word[:i] + c + word[i+1:]
    #                 if new_word in wordList:
    #                     que.append((new_word, step + 1))
    #                     wordList.remove(new_word)
    #     return 0
        
# leetcode submit region end(Prohibit modification and deletion)
