# 班上有 N 名学生。其中有些人是朋友，有些则不是。他们的友谊具有是传递性。如果已知 A 是 B 的朋友，B 是 C 的朋友，那么我们可以认为 A 也是 C 
# 的朋友。所谓的朋友圈，是指所有朋友的集合。 
# 
#  给定一个 N * N 的矩阵 M，表示班级中学生之间的朋友关系。如果M[i][j] = 1，表示已知第 i 个和 j 个学生互为朋友关系，否则为不知道。你
# 必须输出所有学生中的已知的朋友圈总数。 
# 
#  示例 1: 
# 
#  
# 输入: 
# [[1,1,0],
#  [1,1,0],
#  [0,0,1]]
# 输出: 2 
# 说明：已知学生0和学生1互为朋友，他们在一个朋友圈。
# 第2个学生自己在一个朋友圈。所以返回2。
#  
# 
#  示例 2: 
# 
#  
# 输入: 
# [[1,1,0],
#  [1,1,1],
#  [0,1,1]]
# 输出: 1
# 说明：已知学生0和学生1互为朋友，学生1和学生2互为朋友，所以学生0和学生2也是朋友，所以他们三个在一个朋友圈，返回1。
#  
# 
#  注意： 
# 
#  
#  N 在[1,200]的范围内。 
#  对于所有学生，有M[i][i] = 1。 
#  如果有M[i][j] = 1，则有M[j][i] = 1。 
#  
#  Related Topics 深度优先搜索 并查集


# leetcode submit region begin(Prohibit modification and deletion)
class Solution:
    def findCircleNum(self, M: List[List[int]]) -> int:
        # DFS，从学生i开始，访问与其有关系的学生j，进行DFS(
        # def dfs(i):
        #     for j in range(N):
        #         if not j in visited and M[i][j] == 1:
        #             visited.add(j)
        #             dfs(j)
        #
        # if not M: return 0
        # N = len(M)
        # visited, count = set(), 0
        # for i in range(N):
        #     if i not in visited:
        #         dfs(i)
        #         count += 1
        # return count

        # 并查集
        if not M: return 0
        N = len(M)
        p = [i for i in range(N)]
        for i in range(N):
            for j in range(N):
                if M[i][j]:
                    self._union(p, i, j)
        return len(set(self._parent(p, i) for i in range(N)))

    def _union(self, p, i, j):
        pi = self._parent(p, i)
        pj = self._parent(p, j)
        p[pi] = pj

    def _parent(self, p, i):
        root = i
        while root != p[root]:
            root = p[root]
        while p[i] != i:
            x = i
            i = p[i]
            p[x] = root
        return root
# leetcode submit region end(Prohibit modification and deletion)


