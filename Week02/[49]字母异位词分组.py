# 给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。 
# 
#  示例: 
# 
#  输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
# 输出:
# [
#   ["ate","eat","tea"],
#   ["nat","tan"],
#   ["bat"]
# ] 
# 
#  说明： 
# 
#  
#  所有输入均为小写字母。 
#  不考虑答案输出的顺序。 
#  
#  Related Topics 哈希表 字符串


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 1. 哈希表，使用排序后的tuple作为键值
        # O(NKlogK)
        d = collections.defaultdict(list)
        for s in strs:
            d[tuple(sorted(s))].append(s)
        return list(d.values())

        # 2. 使用长度为26的数组来表示
        # O(NK)
        # d = collections.defaultdict(list)
        # for s in strs:
        #     count = [0] * 26
        #     for char in s:
        #         count[ord(char) - ord('a')] += 1
        #     d[tuple(count)].append(s)
        # return list(d.values())
# leetcode submit region end(Prohibit modification and deletion)
