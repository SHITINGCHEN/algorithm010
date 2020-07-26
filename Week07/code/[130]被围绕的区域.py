# 给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。 
# 
#  找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。 
# 
#  示例: 
# 
#  X X X X
# X O O X
# X X O X
# X O X X
#  
# 
#  运行你的函数后，矩阵变为： 
# 
#  X X X X
# X X X X
# X X X X
# X O X X
#  
# 
#  解释: 
# 
#  被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被
# 填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。 
#  Related Topics 深度优先搜索 广度优先搜索 并查集


# leetcode submit region begin(Prohibit modification and deletion)

# class Solution:
#     def solve(self, board: List[List[str]]) -> None:

        # 此题看似找所有'O'的连通区域，
        # 但实际上只要找边界上的'O'的连通区域，保留
        # 非此联通区域的'O'变为'X'即可

        # 思路1，DFS
        # if not board or not board[0]: return board
        # m, n = len(board), len(board[0])
        #
        # def dfs(i, j):
        #     for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        #         x, y = i + dx, j + dy
        #         if 0 <= x < m and 0 <= y < n and board[x][y] == 'O':
        #             board[x][y] = 'B'
        #             dfs(x, y)
        #
        # for i in range(m):
        #     if board[i][0] == 'O':
        #         board[i][0] = 'B'
        #         dfs(i, 0)
        #     if board[i][n - 1] == 'O':
        #         board[i][n - 1] = 'B'
        #         dfs(i, n - 1)
        #
        # for j in range(n):
        #     if board[0][j] == 'O':
        #         board[0][j] = 'B'
        #         dfs(0, j)
        #     if board[m - 1][j] == 'O':
        #         board[m - 1][j] = 'B'
        #         dfs(m - 1, j)
        # # print(board)
        # for i in range(m):
        #     for j in range(n):
        #         if board[i][j] == 'O':
        #             board[i][j] = 'X'
        #         elif board[i][j] == 'B':
        #             board[i][j] = 'O'


class Solution:
    def solve(self, board: List[List[str]]) -> None:
        # 并查集
        if not board or not board[0]: return board
        m, n = len(board), len(board[0])
        uf = UnionFind(m * n + 1)
        dummy_node = m * n # 边界上的O连到dummy node
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    if i==0 or i==m-1 or j==0 or j==n-1:
                        uf.union(i*n + j, dummy_node)
                    else:
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            if board[i + dx][j + dy] == 'O':
                                uf.union(i * n + j, (i + dx) * n + j + dy)
        for i in range(m):
            for j in range(n):
                if uf.find(dummy_node) == uf.find(i * n + j):
                    board[i][j] = 'O'
                else:
                    board[i][j] = 'X'

class UnionFind:
    def __init__(self, n):
        self.parent = [i for i in range(n)]

    def find(self, p):
        while p != self.parent[p]:
            self.parent[p] = self.parent[self.parent[p]]
            p = self.parent[p]
        return p

    def is_connected(self, p, q):
        return self.find(p) == self.find(q)

    def union(self, p, q):
        p_root = self.find(p)
        q_root = self.find(q)
        if p_root != q_root:
            self.parent[p_root] = q_root

# leetcode submit region end(Prohibit modification and deletion)
