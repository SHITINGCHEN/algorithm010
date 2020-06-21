# 给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。 
# 
#  示例 1: 
# 
#  输入: s = "anagram", t = "nagaram"
# 输出: true
#  
# 
#  示例 2: 
# 
#  输入: s = "rat", t = "car"
# 输出: false 
# 
#  说明: 
# 你可以假设字符串只包含小写字母。 
# 
#  进阶: 
# 如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？ 
#  Related Topics 排序 哈希表


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        # 1. 暴力法：sort, 比较sort完是否相等 O(nlogn)
        # s, t = list(s), list(t)
        # s.sort()
        # t.sort()
        # return s==t
        # 2. dict --> 统计频次
        # d = {}
        # for char in s:
        #     if char in d:
        #         d[char] += 1
        #     else:
        #         d[char] = 1
        # for char in t:
        #     if char in d:
        #         d[char] -= 1
        #     else:
        #         return False
        # for key in d:
        #     if d[key] != 0:
        #         return False
        # return True
        # 3. set + count 统计频次
        set_s = set(s)
        set_t = set(t)
        if len(set_s) != len(set_t):
            return False
        for char in set_s:
            if s.count(char) != t.count(char):
                return False
        return True
# leetcode submit region end(Prohibit modification and deletion)
