# 编写一个函数来查找字符串数组中的最长公共前缀。 
# 
#  如果不存在公共前缀，返回空字符串 ""。 
# 
#  示例 1: 
# 
#  输入: ["flower","flow","flight"]
# 输出: "fl"
#  
# 
#  示例 2: 
# 
#  输入: ["dog","racecar","car"]
# 输出: ""
# 解释: 输入不存在公共前缀。
#  
# 
#  说明: 
# 
#  所有输入只包含小写字母 a-z 。 
#  Related Topics 字符串


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        # if not strs: return ''
        # for i in range(len(strs[0])):
        #     c = strs[0][i]
        #     for j in range(1, len(strs)):
        #         if len(strs[j]) == i or strs[j][i] != c:
        #             return strs[0][:i]
        # return strs[0]

        root = dict()
        for s in strs:
            p = root
            for c in s:
                if c not in p:
                    p.setdefault(c, dict())
                p = p.get(c)
            p.setdefault('#')
        p = root
        res = []
        while isinstance(p, dict) and len(p) == 1:
            x, p = p.popitem()
            if x == '#':
                break
            res.append(x)
        return ''.join(res)
# leetcode submit region end(Prohibit modification and deletion)
