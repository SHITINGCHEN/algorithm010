# 给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。 
# 
#  具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被计为是不同的子串。 
# 
#  示例 1: 
# 
#  
# 输入: "abc"
# 输出: 3
# 解释: 三个回文子串: "a", "b", "c".
#  
# 
#  示例 2: 
# 
#  
# 输入: "aaa"
# 输出: 6
# 说明: 6个回文子串: "a", "a", "a", "aa", "aa", "aaa".
#  
# 
#  注意: 
# 
#  
#  输入的字符串长度不会超过1000。 
#  
#  Related Topics 字符串 动态规划


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def countSubstrings(self, s: str) -> int:
        if not s: return 0
        # 二维DP
        # DP[i][j]: 以第i个字符开始，第j个字符结束时的子串是否为回文子串
        # DP方程：已知dp[i+1][j-1] = True 即s[i+1:j-1]是回文子串，那么s[i] == s[j]时，dp[i][j] = True, 反之为false
        dp = [[False] * len(s) for _ in range(len(s))]
        # 初始化
        for i in range(len(s)):
            dp[i][i] = True
        # 因为先计算dp[i+1][j-1]，因此我们从右下角开始算
        count = len(s)
        for i in range(len(s) - 2, -1, -1):
            for j in range(i + 1, len(s)):
                if s[i] == s[j]:
                    # print(i,j)
                    if j == i + 1 or dp[i + 1][j - 1]:
                        dp[i][j] = True
                        count += 1
        return count
# leetcode submit region end(Prohibit modification and deletion)
